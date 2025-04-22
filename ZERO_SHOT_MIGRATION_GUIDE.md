# Zero-Shot Learning Migration Guide for InternVL_V2

## Overview

This guide outlines the steps to integrate zero-shot learning capabilities into the InternVL_V2 codebase. Zero-shot learning enables the model to classify new types of documents or answer questions about documents without requiring any training examples for these new categories.

## What is Zero-Shot Learning?

Zero-shot learning is a machine learning paradigm that aims to recognize objects or classes that weren't seen during training. It leverages semantic descriptions of classes and transfers knowledge from seen to unseen classes, typically using shared semantic space between visual features and textual descriptions.

## Benefits for InternVL_V2

- **Handle New Document Types**: Classify new document types without retraining
- **Flexible Question Answering**: Answer new types of questions not seen during training
- **Reduced Annotation Requirements**: Eliminate the need for labeled examples of every document type
- **Extended Functionality**: Identify and extract information based on textual descriptions alone
- **Future-Proof System**: Adapt to new document formats and requirements with minimal effort

## Migration Steps

### 1. Implement Vision-Language Alignment Module

Create a new module that aligns visual features with textual descriptions:

```python
# Path: models/vision_language/alignment.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionLanguageAlignment(nn.Module):
    """
    Module for aligning visual and language features in a common embedding space.
    This enables zero-shot classification and retrieval.
    """
    
    def __init__(
        self, 
        vision_dim: int,
        language_dim: int,
        embedding_dim: int = 512,
        temperature: float = 0.07
    ):
        super().__init__()
        
        # Projection layers for vision and language features
        self.vision_projection = nn.Linear(vision_dim, embedding_dim)
        self.language_projection = nn.Linear(language_dim, embedding_dim)
        
        # Layer normalization to stabilize embeddings
        self.vision_ln = nn.LayerNorm(embedding_dim)
        self.language_ln = nn.LayerNorm(embedding_dim)
        
        # Temperature parameter for scaling logits
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize projection matrices with orthogonal initialization."""
        nn.init.orthogonal_(self.vision_projection.weight)
        nn.init.orthogonal_(self.language_projection.weight)
        nn.init.zeros_(self.vision_projection.bias)
        nn.init.zeros_(self.language_projection.bias)
    
    def encode_image(self, image_features):
        """Project and normalize image features."""
        image_embeddings = self.vision_projection(image_features)
        image_embeddings = self.vision_ln(image_embeddings)
        return F.normalize(image_embeddings, dim=-1)
    
    def encode_text(self, text_features):
        """Project and normalize text features."""
        text_embeddings = self.language_projection(text_features)
        text_embeddings = self.language_ln(text_embeddings)
        return F.normalize(text_embeddings, dim=-1)
    
    def forward(self, image_features, text_features):
        """
        Compute similarity between image and text features.
        
        Args:
            image_features: Tensor of shape [batch_size, vision_dim]
            text_features: Tensor of shape [batch_size, language_dim]
            
        Returns:
            similarity: Tensor of shape [batch_size, batch_size]
        """
        # Project and normalize features
        image_embeddings = self.encode_image(image_features)
        text_embeddings = self.encode_text(text_features)
        
        # Compute cosine similarity
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        
        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings, 
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text
        }
```

### 2. Create Zero-Shot Document Classifier

Implement a zero-shot document classification module that can classify documents based on textual class descriptions:

```python
# Path: models/vision_language/zero_shot_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from models.vision_language.internvl2 import InternVL2ReceiptClassifier
from models.vision_language.alignment import VisionLanguageAlignment

class ZeroShotDocumentClassifier(nn.Module):
    """
    Zero-shot document classifier that uses textual descriptions 
    to classify images without requiring class-specific training examples.
    """
    
    def __init__(
        self, 
        config: Dict,
        class_descriptions: Dict[str, str] = None
    ):
        super().__init__()
        
        # Initialize vision encoder based on InternVL2
        self.vision_model = InternVL2ReceiptClassifier(config, pretrained=True)
        
        # Get vision encoder output dimension
        vision_dim = self.vision_model.vision_encoder.config.hidden_size
        
        # Initialize language model for processing class descriptions
        from transformers import AutoModel, AutoTokenizer
        
        language_model_name = config.get("zero_shot", {}).get(
            "language_model", "all-mpnet-base-v2"
        )
        self.text_encoder = AutoModel.from_pretrained(language_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        
        # Get language encoder output dimension
        language_dim = self.text_encoder.config.hidden_size
        
        # Create vision-language alignment module
        embedding_dim = config.get("zero_shot", {}).get("embedding_dim", 512)
        temperature = config.get("zero_shot", {}).get("temperature", 0.07)
        
        self.alignment = VisionLanguageAlignment(
            vision_dim=vision_dim,
            language_dim=language_dim,
            embedding_dim=embedding_dim,
            temperature=temperature
        )
        
        # Store class descriptions if provided
        self.class_descriptions = class_descriptions
        self.class_embeddings = None
        
        # If class descriptions are provided, precompute their embeddings
        if class_descriptions:
            self._precompute_class_embeddings()
    
    def _precompute_class_embeddings(self):
        """Precompute embeddings for all class descriptions."""
        self.text_encoder.eval()
        self.alignment.eval()
        
        # Get sorted class names to ensure consistent ordering
        class_names = sorted(self.class_descriptions.keys())
        descriptions = [self.class_descriptions[name] for name in class_names]
        
        # Tokenize descriptions
        batch_tokens = self.tokenizer(
            descriptions, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Encode descriptions
        with torch.no_grad():
            # Move tensors to device
            device = next(self.text_encoder.parameters()).device
            input_ids = batch_tokens.input_ids.to(device)
            attention_mask = batch_tokens.attention_mask.to(device)
            
            # Get text features
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get mean of last hidden state as text features
            text_features = outputs.last_hidden_state[:, 0, :]
            
            # Project and normalize features
            text_embeddings = self.alignment.encode_text(text_features)
        
        # Store embeddings and class names
        self.class_embeddings = text_embeddings
        self.class_names = class_names
    
    def set_class_descriptions(self, class_descriptions: Dict[str, str]):
        """
        Set or update class descriptions.
        
        Args:
            class_descriptions: Dictionary mapping class names to descriptions
        """
        self.class_descriptions = class_descriptions
        self._precompute_class_embeddings()
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode an image using the vision encoder.
        
        Args:
            pixel_values: Tensor of shape [batch_size, 3, H, W]
            
        Returns:
            image_embeddings: Tensor of shape [batch_size, embedding_dim]
        """
        # Get vision features
        outputs = self.vision_model(pixel_values)
        
        # Use the pooled embeddings
        vision_features = outputs["embeddings"]
        
        # Project and normalize features
        image_embeddings = self.alignment.encode_image(vision_features)
        
        return image_embeddings
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text descriptions.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            text_embeddings: Tensor of shape [len(texts), embedding_dim]
        """
        # Tokenize texts
        batch_tokens = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.text_encoder.parameters()).device
        input_ids = batch_tokens.input_ids.to(device)
        attention_mask = batch_tokens.attention_mask.to(device)
        
        # Get text features
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get mean of last hidden state as text features
        text_features = outputs.last_hidden_state[:, 0, :]
        
        # Project and normalize features
        text_embeddings = self.alignment.encode_text(text_features)
        
        return text_embeddings
    
    def classify(
        self, 
        pixel_values: torch.Tensor,
        custom_descriptions: Dict[str, str] = None
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Classify images using zero-shot classification.
        
        Args:
            pixel_values: Tensor of shape [batch_size, 3, H, W]
            custom_descriptions: Optional dictionary for one-off classification
                                without updating stored class descriptions
        
        Returns:
            predicted_classes: List of predicted class names
            confidence_scores: Tensor of confidence scores
        """
        # Encode image
        image_embeddings = self.encode_image(pixel_values)
        
        # Use custom descriptions if provided
        if custom_descriptions:
            class_names = sorted(custom_descriptions.keys())
            descriptions = [custom_descriptions[name] for name in class_names]
            text_embeddings = self.encode_text(descriptions)
        else:
            # Use precomputed class embeddings
            if self.class_embeddings is None:
                raise ValueError("No class descriptions provided. Call set_class_descriptions first.")
            text_embeddings = self.class_embeddings
            class_names = self.class_names
        
        # Compute similarity scores
        similarity = torch.matmul(image_embeddings, text_embeddings.t()) / self.alignment.temperature
        
        # Get predictions
        max_scores, max_indices = similarity.max(dim=1)
        predictions = [class_names[idx] for idx in max_indices.cpu().tolist()]
        
        # Convert to probabilities
        probabilities = F.softmax(similarity, dim=1)
        
        return predictions, probabilities
    
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        text_inputs: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training the alignment module.
        
        Args:
            pixel_values: Tensor of shape [batch_size, 3, H, W]
            text_inputs: List of text descriptions
            
        Returns:
            Dictionary containing logits and embeddings
        """
        # Get vision features
        vision_outputs = self.vision_model(pixel_values)
        vision_features = vision_outputs["embeddings"]
        
        # Get text features
        if text_inputs:
            # Tokenize text inputs
            batch_tokens = self.tokenizer(
                text_inputs, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            
            # Move to device
            device = next(self.text_encoder.parameters()).device
            input_ids = batch_tokens.input_ids.to(device)
            attention_mask = batch_tokens.attention_mask.to(device)
            
            # Get text features
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Use pooled output or first token
            text_features = text_outputs.last_hidden_state[:, 0, :]
        else:
            # If no text inputs, assume we're using precomputed class embeddings
            # Just return vision features
            return {
                "vision_features": vision_features
            }
        
        # Compute alignment
        alignment_outputs = self.alignment(vision_features, text_features)
        
        return {
            **alignment_outputs,
            "vision_features": vision_features,
            "text_features": text_features
        }
```

### 3. Implement Zero-Shot Loss Functions

Create specialized loss functions for training the zero-shot model:

```python
# Path: utils/zero_shot_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training vision-language alignment.
    
    This is a symmetric loss that pulls matching image-text pairs together
    while pushing non-matching pairs apart.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits_per_image, logits_per_text, labels=None):
        """
        Compute contrastive loss.
        
        Args:
            logits_per_image: Tensor of shape [batch_size, batch_size]
            logits_per_text: Tensor of shape [batch_size, batch_size]
            labels: Optional target labels. If None, diagonal is used.
            
        Returns:
            Loss value
        """
        # If labels not provided, assume diagonal (identity matrix) as ground truth
        if labels is None:
            batch_size = logits_per_image.size(0)
            labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # Image-to-text loss (match each image to its text)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        
        # Text-to-image loss (match each text to its image)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        # Average the losses for a symmetric contrastive loss
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss

class ZeroShotClassificationLoss(nn.Module):
    """
    Loss function for training zero-shot classification.
    
    Combines contrastive loss with classification loss to improve
    zero-shot performance on document types.
    """
    
    def __init__(self, num_classes, temperature=0.07, weight=None):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.classification_loss = nn.CrossEntropyLoss(weight=weight)
        self.num_classes = num_classes
    
    def forward(self, outputs, targets=None):
        """
        Compute loss for zero-shot classification.
        
        Args:
            outputs: Dictionary from ZeroShotDocumentClassifier
            targets: Optional classification targets
            
        Returns:
            Loss value
        """
        # Contrastive loss between image and text
        contrastive = self.contrastive_loss(
            outputs["logits_per_image"],
            outputs["logits_per_text"]
        )
        
        # Return only contrastive loss if no targets
        if targets is None:
            return contrastive
        
        # Compute classification loss if targets are provided
        logits = outputs["logits_per_image"]
        classification = self.classification_loss(logits, targets)
        
        # Combine losses (equal weighting)
        loss = contrastive + classification
        
        return loss
```

### 4. Create Zero-Shot Prompt Templates

Create a template system to generate effective textual descriptions for zero-shot learning:

```python
# Path: utils/zero_shot_templates.py
from typing import Dict, List, Union, Optional

class ZeroShotTemplates:
    """
    Generate templates for zero-shot learning prompts.
    
    Templates are used to convert class names into rich textual descriptions
    that maximize zero-shot performance.
    """
    
    def __init__(self):
        # Document type templates
        self.document_templates = [
            "a photo of a {class_name}",
            "an image of a {class_name} document",
            "a scanned {class_name}",
            "a business document: {class_name}",
            "a financial document of type: {class_name}",
            "a document that would be classified as a {class_name}"
        ]
        
        # Templates for specific document categories
        self.receipt_templates = [
            "a receipt from a store or business",
            "a shopping receipt with items and prices",
            "a payment receipt with transaction details",
            "a receipt showing purchased items",
            "a paper receipt with total amount paid",
            "a store receipt with date and payment information"
        ]
        
        self.tax_document_templates = [
            "an official tax document with financial information",
            "a tax form with personal details and calculations",
            "an ATO tax document with TFN and tax year",
            "a government tax form with financial data",
            "an official tax statement with income details",
            "a formal tax document with payment information"
        ]
        
        self.invoice_templates = [
            "a business invoice for services or products",
            "an invoice with company details and amounts",
            "a formal invoice with itemized costs",
            "a bill for services rendered with payment details",
            "an invoice requesting payment for products",
            "a commercial invoice with company letterhead"
        ]
    
    def get_templates(self, category: str) -> List[str]:
        """
        Get templates for a specific document category.
        
        Args:
            category: Document category (e.g., 'receipt', 'tax_document', 'invoice')
            
        Returns:
            List of templates for the category
        """
        if category == "receipt":
            return self.receipt_templates
        elif category == "tax_document":
            return self.tax_document_templates
        elif category == "invoice":
            return self.invoice_templates
        else:
            # Default to general document templates
            return self.document_templates
    
    def create_class_descriptions(
        self, 
        classes: List[str], 
        use_multiple_templates: bool = True,
        custom_templates: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, str]:
        """
        Create rich textual descriptions for classes.
        
        Args:
            classes: List of class names
            use_multiple_templates: Whether to combine multiple templates
            custom_templates: Optional dictionary mapping class names to custom templates
            
        Returns:
            Dictionary mapping class names to descriptions
        """
        descriptions = {}
        
        for class_name in classes:
            # Determine which templates to use
            if custom_templates and class_name in custom_templates:
                templates = custom_templates[class_name]
            else:
                # Try to match class name to a category
                if "receipt" in class_name.lower():
                    templates = self.receipt_templates
                elif "tax" in class_name.lower():
                    templates = self.tax_document_templates
                elif "invoice" in class_name.lower():
                    templates = self.invoice_templates
                else:
                    templates = self.document_templates
            
            if use_multiple_templates:
                # Combine multiple templates for a richer description
                description = ". ".join([
                    template.format(class_name=class_name) 
                    for template in templates[:3]  # Use top 3 templates
                ])
            else:
                # Use just the first template
                description = templates[0].format(class_name=class_name)
            
            descriptions[class_name] = description
        
        return descriptions
```

### 5. Data Loaders for Zero-Shot Training

Create a specialized data loader for training the zero-shot model:

```python
# Path: data/datasets/zero_shot_dataset.py
import os
import json
from typing import Dict, List, Optional, Tuple, Union
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class ZeroShotDataset(Dataset):
    """
    Dataset for training zero-shot models with paired image-text data.
    """
    
    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        transform = None,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Directory containing images
            captions_file: JSON file with image-caption pairs
            transform: Image transformations to apply
            max_samples_per_class: Maximum samples per class (for balanced sampling)
        """
        self.image_dir = image_dir
        
        # Load captions
        with open(captions_file, 'r') as f:
            self.captions_data = json.load(f)
        
        # Setup transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Create image-caption pairs
        self.samples = []
        self.classes = set()
        
        # Keep track of samples per class
        class_counts = {}
        
        for item in self.captions_data:
            image_name = item["image"]
            caption = item["caption"]
            class_name = item.get("class", "unknown")
            
            # Track classes
            self.classes.add(class_name)
            
            # Count samples per class
            if class_name not in class_counts:
                class_counts[class_name] = 0
            
            # Skip if we've reached max samples for this class
            if max_samples_per_class and class_counts[class_name] >= max_samples_per_class:
                continue
            
            class_counts[class_name] += 1
            
            # Add sample
            self.samples.append({
                "image": os.path.join(self.image_dir, image_name),
                "caption": caption,
                "class": class_name
            })
        
        # Convert classes to a sorted list
        self.class_list = sorted(list(self.classes))
        
        # Create a mapping from class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load and transform image
        image = Image.open(sample["image"]).convert("RGB")
        image = self.transform(image)
        
        # Get caption and class
        caption = sample["caption"]
        class_name = sample["class"]
        class_idx = self.class_to_idx[class_name]
        
        return {
            "pixel_values": image,
            "caption": caption,
            "class": class_name,
            "class_idx": class_idx
        }
    
    def get_class_names(self):
        """Get all class names in the dataset."""
        return self.class_list
```

### 6. Zero-Shot Training Scripts

Create a script for training the zero-shot model:

```python
# Path: scripts/training/train_zero_shot.py
import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data.datasets.zero_shot_dataset import ZeroShotDataset
from models.vision_language.zero_shot_classifier import ZeroShotDocumentClassifier
from utils.zero_shot_loss import ZeroShotClassificationLoss
from utils.zero_shot_templates import ZeroShotTemplates
from utils.device import get_device
from utils.reproducibility import set_seed

def train_zero_shot_model(
    config_path: str,
    image_dir: str,
    captions_file: str,
    output_dir: str,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    device = None
):
    """
    Train a zero-shot document classifier.
    
    Args:
        config_path: Path to model config file
        image_dir: Directory containing training images
        captions_file: JSON file with image captions
        output_dir: Output directory for saving the model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on
    """
    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "training.log"))
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Set device
    if device is None:
        device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure zero_shot config exists
    if "zero_shot" not in config:
        config["zero_shot"] = {}
    
    # Create dataset
    dataset = ZeroShotDataset(
        image_dir=image_dir,
        captions_file=captions_file
    )
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create template generator for class descriptions
    template_generator = ZeroShotTemplates()
    
    # Get class names from dataset
    class_names = dataset.get_class_names()
    
    # Create rich class descriptions
    class_descriptions = template_generator.create_class_descriptions(
        class_names, use_multiple_templates=True
    )
    
    # Log class descriptions
    logger.info(f"Generated descriptions for {len(class_descriptions)} classes:")
    for class_name, description in class_descriptions.items():
        logger.info(f"  {class_name}: {description}")
    
    # Create model
    model = ZeroShotDocumentClassifier(
        config=config,
        class_descriptions=class_descriptions
    )
    model.to(device)
    
    # Create loss function
    num_classes = len(class_names)
    loss_fn = ZeroShotClassificationLoss(num_classes=num_classes)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            captions = [str(caption) for caption in batch["caption"]]
            class_indices = batch["class_idx"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(pixel_values, captions)
            
            # Compute loss
            loss = loss_fn(outputs, class_indices)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average training loss
        train_loss /= len(train_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                pixel_values = batch["pixel_values"].to(device)
                captions = [str(caption) for caption in batch["caption"]]
                class_indices = batch["class_idx"].to(device)
                
                # Forward pass
                outputs = model(pixel_values, captions)
                
                # Compute loss
                loss = loss_fn(outputs, class_indices)
                
                # Update metrics
                val_loss += loss.item()
                
                # Compute accuracy
                logits = outputs["logits_per_image"]
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == class_indices).sum().item()
                total += class_indices.size(0)
        
        # Average validation loss and accuracy
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model
            save_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'class_names': class_names,
                'class_descriptions': class_descriptions
            }, save_path)
            
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
    
    # Save final model
    save_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'class_names': class_names,
        'class_descriptions': class_descriptions
    }, save_path)
    
    logger.info(f"Final model saved to {save_path}")
    logger.info("Training completed!")
    
    return {
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train zero-shot document classifier")
    parser.add_argument("--config", type=str, default="config/model/zero_shot_config.yaml", help="Path to model config")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--captions-file", type=str, required=True, help="JSON file with image captions")
    parser.add_argument("--output-dir", type=str, default="models/zero_shot", help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Train model
    train_zero_shot_model(
        config_path=args.config,
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()
```

### 7. Zero-Shot Inference Scripts

Create a script for zero-shot inference:

```python
# Path: scripts/inference/zero_shot_inference.py
import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from models.vision_language.zero_shot_classifier import ZeroShotDocumentClassifier
from utils.zero_shot_templates import ZeroShotTemplates
from utils.device import get_device

def load_zero_shot_model(model_path: str, config_path: str, device) -> ZeroShotDocumentClassifier:
    """
    Load pretrained zero-shot classifier.
    
    Args:
        model_path: Path to saved model
        config_path: Path to model config
        device: Device to run on
        
    Returns:
        Loaded model
    """
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load saved state
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract class descriptions
    class_descriptions = checkpoint.get('class_descriptions', {})
    
    # Create model
    model = ZeroShotDocumentClassifier(
        config=config,
        class_descriptions=class_descriptions
    )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess an image for model input.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image tensor
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def zero_shot_classify(
    model: ZeroShotDocumentClassifier,
    image_path: str,
    custom_classes: Optional[List[str]] = None,
    device = None
) -> Dict:
    """
    Perform zero-shot classification on an image.
    
    Args:
        model: Zero-shot classifier model
        image_path: Path to image file
        custom_classes: Optional list of custom classes to use instead of model's classes
        device: Device to run on
        
    Returns:
        Dictionary with classification results
    """
    # Set device
    if device is None:
        device = next(model.parameters()).device
    
    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)
    
    # Generate custom class descriptions if needed
    if custom_classes:
        template_generator = ZeroShotTemplates()
        class_descriptions = template_generator.create_class_descriptions(custom_classes)
        
        # Classify with custom classes
        predictions, probabilities = model.classify(
            image_tensor, custom_descriptions=class_descriptions
        )
    else:
        # Classify with model's built-in classes
        predictions, probabilities = model.classify(image_tensor)
    
    # Get prediction for single image
    prediction = predictions[0]
    probs = probabilities[0].cpu().tolist()
    
    # Create result dictionary
    if custom_classes:
        # Map probabilities to class names
        confidence_scores = {cls: prob for cls, prob in zip(custom_classes, probs)}
    else:
        # Use model's class names
        class_names = model.class_names
        confidence_scores = {cls: prob for cls, prob in zip(class_names, probs)}
    
    result = {
        "prediction": prediction,
        "confidence_scores": confidence_scores
    }
    
    return result

def main():
    """Main function for zero-shot inference."""
    parser = argparse.ArgumentParser(description="Zero-shot document classification")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--output", type=str, default="zero_shot_results.json", help="Output file for results")
    parser.add_argument("--custom-classes", type=str, default=None, help="Optional JSON file with custom classes to classify")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # Set device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_zero_shot_model(args.model, args.config, device)
    
    # Load custom classes if provided
    custom_classes = None
    if args.custom_classes:
        logger.info(f"Loading custom classes from {args.custom_classes}")
        with open(args.custom_classes, 'r') as f:
            custom_classes = json.load(f)
    
    # Classify image
    logger.info(f"Classifying image: {args.image}")
    result = zero_shot_classify(model, args.image, custom_classes, device)
    
    # Log prediction
    logger.info(f"Prediction: {result['prediction']}")
    logger.info("Confidence scores:")
    for cls, score in sorted(result['confidence_scores'].items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {cls}: {score:.4f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
```

### 8. Zero-Shot Question Answering

Implement zero-shot question answering capabilities:

```python
# Path: models/vision_language/zero_shot_qa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.vision_language.internvl2 import InternVL2MultimodalModel

class ZeroShotQuestionAnswering(nn.Module):
    """
    Zero-shot question answering for documents.
    
    Uses InternVL2 with enhanced prompting to answer questions
    about documents without specific training examples.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Initialize InternVL2 multimodal model
        self.model = InternVL2MultimodalModel(config, pretrained=True)
        
        # Question type templates for zero-shot prompting
        self.question_templates = {
            "document_type": [
                "What type of document is this?",
                "Is this a receipt, invoice, or tax document?",
                "Identify the type of document shown."
            ],
            "counting": [
                "How many receipts are in this image?",
                "Count the number of documents shown.",
                "How many separate receipts can you see?"
            ],
            "extraction": [
                "What is the total amount on this receipt?",
                "What date is shown on this document?",
                "What is the store name on this receipt?"
            ],
            "verification": [
                "Does this receipt contain item X?",
                "Is the total amount greater than $100?",
                "Was this purchase made in 2023?"
            ]
        }
        
        # Answer templates for zero-shot generation
        self.answer_templates = {
            "document_type": [
                "This is a {document_type}.",
                "The document type is {document_type}.",
                "This document is classified as a {document_type}."
            ],
            "counting": [
                "There are {count} receipts in this image.",
                "I can see {count} separate documents.",
                "The image contains {count} receipts."
            ],
            "extraction": [
                "The {entity_type} is {entity_value}.",
                "Based on the document, the {entity_type} is {entity_value}.",
                "I found the {entity_type}: {entity_value}."
            ],
            "verification": [
                "Yes, {verification_detail}.",
                "No, {verification_detail}.",
                "I cannot verify this from the document."
            ]
        }
    
    def classify_question(self, question: str) -> str:
        """
        Determine question type for zero-shot prompting.
        
        Args:
            question: Question text
            
        Returns:
            Question type
        """
        question = question.lower()
        
        # Simple rule-based classification
        if any(keyword in question for keyword in ["how many", "count", "number of"]):
            return "counting"
        elif any(keyword in question for keyword in ["what type", "document type", "kind of document"]):
            return "document_type"
        elif any(keyword in question for keyword in ["what is", "extract", "find", "identify"]):
            return "extraction"
        elif any(keyword in question for keyword in ["is there", "does it", "verify", "confirm"]):
            return "verification"
        else:
            # Default to extraction for unknown question types
            return "extraction"
    
    def generate_enhanced_prompt(self, question: str) -> str:
        """
        Generate an enhanced prompt for zero-shot question answering.
        
        Args:
            question: Original question
            
        Returns:
            Enhanced prompt with zero-shot instructions
        """
        # Classify question type
        question_type = self.classify_question(question)
        
        # Get examples for this question type
        examples = self.question_templates.get(question_type, [])
        
        # Create enhanced prompt with few-shot examples
        prompt = "Answer the following question about the document image:\n\n"
        
        # Add examples if available
        if examples:
            prompt += "Here are similar questions:\n"
            for example in examples[:2]:  # Limit to 2 examples
                prompt += f"- {example}\n"
            
            prompt += "\nNow please answer this question:\n"
        
        # Add the actual question
        prompt += f"{question}"
        
        return prompt
    
    def answer_question(
        self, 
        image: torch.Tensor, 
        question: str,
        max_length: int = 50
    ) -> str:
        """
        Answer a question about an image in a zero-shot manner.
        
        Args:
            image: Image tensor [1, 3, H, W]
            question: Question string
            max_length: Maximum response length
            
        Returns:
            Generated answer text
        """
        # Generate enhanced prompt
        enhanced_question = self.generate_enhanced_prompt(question)
        
        # Prepare inputs for model
        inputs = self.model.prepare_inputs(
            images=image,
            text_prompts=[enhanced_question]
        )
        
        # Generate response
        with torch.no_grad():
            generated_ids, responses = self.model.generate_response(
                pixel_values=inputs["pixel_values"],
                text_input_ids=inputs["text_input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length
            )
        
        # Return the first (only) response
        return responses[0]

def create_zero_shot_qa(config_path: str) -> ZeroShotQuestionAnswering:
    """
    Create a zero-shot QA model.
    
    Args:
        config_path: Path to model config
        
    Returns:
        ZeroShotQuestionAnswering model
    """
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = ZeroShotQuestionAnswering(config)
    
    return model
```

### 9. Configuration Updates

Create a new configuration file for zero-shot learning:

```yaml
# Path: config/model/zero_shot_config.yaml
model:
  # Base model configuration
  pretrained_path: "/Users/tod/PretrainedLLM/InternVL2_5-1B"
  multimodal: true
  
  # Zero-shot specific configuration
  zero_shot:
    enabled: true
    embedding_dim: 512
    temperature: 0.07
    language_model: "all-mpnet-base-v2"
    freeze_vision_encoder: true
    use_cross_attention: true
  
  # Classifier components (used if zero_shot.enabled is False)
  classifier:
    activation: "gelu"
    batch_norm: true
    dropout_rates:
      - 0.2
      - 0.1
    hidden_dims:
      - 512
      - 256
  
  # Model architecture settings
  num_classes: 3

training:
  # Zero-shot training configuration
  zero_shot:
    temperature: 0.07
    contrastive_weight: 1.0
    classification_weight: 1.0
    caption_augmentation: true
    min_captions_per_image: 3
  
  # Standard training configuration
  epochs: 20
  learning_rate: 1.0e-5
  gradient_clip: 1.0
  weight_decay: 0.0001
  batch_size: 32
  optimizer:
    name: "adamw"
    learning_rate: 2.0e-5
    weight_decay: 0.01
  
  early_stopping:
    patience: 5
    min_delta: 0.001
  
  scheduler:
    name: "cosine"
    warmup_steps: 500

# Data configuration
data:
  captions_file: "data/zero_shot/document_captions.json"
  image_dir: "data/zero_shot/images"
  val_split: 0.2
  test_split: 0.1
  max_samples_per_class: null  # Set to an integer to limit samples per class
```

### 10. Zero-Shot Documentation Generation

Create a script to generate documentation for zero-shot classes:

```python
# Path: scripts/utils/generate_zero_shot_docs.py
import argparse
import json
import os
from pathlib import Path
import logging
import sys

from utils.zero_shot_templates import ZeroShotTemplates

def generate_class_documentation(
    classes_file: str,
    output_file: str,
    examples_per_class: int = 5
):
    """
    Generate rich textual documentation for zero-shot classes.
    
    Args:
        classes_file: JSON file with class names
        output_file: Output file for documentation
        examples_per_class: Number of example descriptions per class
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    
    # Load class names
    with open(classes_file, 'r') as f:
        class_data = json.load(f)
    
    # Extract class names
    if isinstance(class_data, dict):
        class_names = list(class_data.keys())
    else:
        class_names = class_data
    
    logger.info(f"Generating documentation for {len(class_names)} classes")
    
    # Create template generator
    template_generator = ZeroShotTemplates()
    
    # Generate documentation
    documentation = {
        "class_descriptions": {},
        "zero_shot_examples": {}
    }
    
    # Generate rich descriptions for each class
    for class_name in class_names:
        # Get the most general description
        description = template_generator.create_class_descriptions(
            [class_name], use_multiple_templates=True
        )[class_name]
        
        # Store in documentation
        documentation["class_descriptions"][class_name] = description
        
        # Generate example captions
        category = "receipt" if "receipt" in class_name.lower() else \
                  "tax_document" if "tax" in class_name.lower() else \
                  "invoice" if "invoice" in class_name.lower() else \
                  "document"
        
        templates = template_generator.get_templates(category)
        examples = []
        
        # Use different templates to generate examples
        for template in templates[:examples_per_class]:
            example = template.format(class_name=class_name)
            examples.append(example)
        
        documentation["zero_shot_examples"][class_name] = examples
    
    # Save documentation
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(documentation, f, indent=2)
    
    logger.info(f"Documentation saved to {output_file}")
    
    return documentation

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate zero-shot class documentation")
    parser.add_argument("--classes", type=str, required=True, help="JSON file with class names")
    parser.add_argument("--output", type=str, default="data/zero_shot/class_documentation.json", help="Output file")
    parser.add_argument("--examples", type=int, default=5, help="Number of examples per class")
    
    args = parser.parse_args()
    
    # Generate documentation
    generate_class_documentation(
        classes_file=args.classes,
        output_file=args.output,
        examples_per_class=args.examples
    )

if __name__ == "__main__":
    main()
```

### 11. Troubleshooting and Debugging Guide

#### Common Issues and Solutions

1. **Poor Zero-Shot Transfer**
   - **Cause**: Insufficient alignment between visual and textual features
   - **Solution**: 
     - Increase embedding dimension in `zero_shot_config.yaml`
     - Improve class descriptions with more detailed templates
     - Use domain-specific language model like "all-mpnet-base-v2"

2. **Out of Memory During Training**
   - **Cause**: Large batch size or model
   - **Solution**:
     - Reduce batch size
     - Enable gradient checkpointing in config
     - Use a smaller language model for text encoding

3. **Low Accuracy on Unseen Classes**
   - **Cause**: Domain gap between training and test classes
   - **Solution**:
     - Add more diverse training data
     - Improve class descriptions with domain-specific terminology
     - Reduce temperature parameter for more confident predictions

4. **Slow Inference**
   - **Cause**: Processing multiple class descriptions at inference time
   - **Solution**:
     - Pre-compute and cache class embedding vectors
     - Reduce number of templates per class
     - Use a lighter weight language model for inference

### 12. Testing the Implementation

To verify your zero-shot implementation:

1. **Generate Class Documentation**:
   ```bash
   PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/internvl_env/bin/python scripts/utils/generate_zero_shot_docs.py --classes data/document_classes.json --output data/zero_shot/class_docs.json
   ```

2. **Train Zero-Shot Model**:
   ```bash
   PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/internvl_env/bin/python scripts/training/train_zero_shot.py --config config/model/zero_shot_config.yaml --image-dir data/documents/images --captions-file data/documents/captions.json --output-dir models/zero_shot
   ```

3. **Test Zero-Shot Classification**:
   ```bash
   PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/internvl_env/bin/python scripts/inference/zero_shot_inference.py --model models/zero_shot/best_model.pt --config config/model/zero_shot_config.yaml --image test_image.jpg --custom-classes data/new_document_types.json
   ```

4. **Test Zero-Shot QA**:
   ```bash
   PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/internvl_env/bin/python scripts/inference/zero_shot_qa.py --config config/model/zero_shot_config.yaml --image test_image.jpg --question "What's the total amount on this receipt?"
   ```

### 13. Performance Optimization Tips

1. **Speed Optimizations**
   - Cache class embeddings during inference
   - Use smaller language models for text encoding
   - Implement batched processing for multiple images

2. **Memory Optimizations**
   - Use gradient checkpointing during training
   - Enable mixed precision training with FP16
   - Freeze vision encoder weights

3. **Accuracy Optimizations**
   - Create domain-specific class descriptions
   - Fine-tune on a small set of in-domain examples
   - Ensemble multiple models for robust predictions

4. **Deployment Optimizations**
   - Quantize model weights for faster inference
   - Use ONNX Runtime for optimized deployment
   - Implement response caching for common queries

## Conclusion

This migration guide provides a comprehensive approach to integrating zero-shot learning capabilities into the InternVL_V2 codebase. By implementing these changes, your system will be able to:

1. **Classify new document types** without requiring labeled examples
2. **Extract information from previously unseen document formats**
3. **Answer novel questions** about documents using zero-shot prompting
4. **Adapt dynamically to new requirements** with just textual descriptions

Zero-shot learning significantly enhances the flexibility and adaptability of document processing systems, allowing them to handle new document types and answer new question types with minimal human intervention.

The implementation uses a combination of:
- **Vision-language alignment** to create a common semantic space
- **Rich textual descriptions** to define classes and tasks
- **Contrastive learning** to align visual and textual features
- **Enhanced prompting techniques** for zero-shot question answering

Follow this guide to transform your InternVL_V2 system into a more flexible, adaptable, and powerful document understanding platform with true zero-shot capabilities.