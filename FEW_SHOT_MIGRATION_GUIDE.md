# Migration Guide: Single Shot Learning for InternVL_V2

## Overview

This document outlines the migration process to adapt the InternVL_V2 codebase for single-shot learning capabilities. Single-shot learning will enable the model to learn from just one or very few examples per class, significantly reducing the data requirements while maintaining performance.

## What is Single-Shot Learning?

Single-shot learning is a machine learning approach where a model aims to recognize new classes from just one or a few training examples. This contrasts with traditional deep learning approaches that typically require large datasets with many examples per class.

## Benefits for InternVL_V2

- **Reduced Data Requirements**: Train on minimal examples per receipt type
- **Faster Adaptation**: Quickly adapt to new document formats with minimal examples
- **Better Generalization**: Learn transferable features that apply across similar documents
- **Resource Efficiency**: Less data collection, annotation, and storage needs

## Migration Steps

### 1. Model Architecture Updates

#### 1.1. Implement Siamese Network Architecture

```python
# Path: models/vision_language/siamese_network.py
import torch
import torch.nn as nn
from models.vision_language.internvl2 import InternVL2ReceiptClassifier

class SiameseInternVL2(nn.Module):
    """Siamese network adapting InternVL2 for single-shot learning."""
    
    def __init__(self, config):
        super().__init__()
        # Initialize with pretrained weights
        self.encoder = InternVL2ReceiptClassifier(config, pretrained=True)
        # Remove the original classification head
        del self.encoder.classification_head
        
        # Add contrastive learning components
        self.embedding_dim = 512
        self.embedding_projection = nn.Linear(
            self.encoder.vision_encoder.config.hidden_size, 
            self.embedding_dim
        )
        
        # Add L2 normalization layer for embedding normalization
        self.l2_normalization = lambda x: nn.functional.normalize(x, p=2, dim=1)
        
    def forward_one(self, x):
        """Process a single image through the network."""
        # Get vision embeddings without classification head
        outputs = self.encoder.vision_encoder(pixel_values=x)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Project to embedding space and normalize
        embedding = self.embedding_projection(pooled_output)
        normalized_embedding = self.l2_normalization(embedding)
        
        return normalized_embedding
    
    def forward(self, x1, x2=None):
        """
        Forward pass with either one input (inference) or two inputs (training).
        
        Args:
            x1: First image
            x2: Second image (optional, for training)
            
        Returns:
            If training (x2 is provided): embeddings for both images
            If inference (x2 is None): embedding for x1 only
        """
        emb1 = self.forward_one(x1)
        
        if x2 is not None:
            emb2 = self.forward_one(x2)
            return emb1, emb2
        
        return emb1
```

#### 1.2. Implement Prototypical Network Adaptation

```python
# Path: models/vision_language/prototypical_network.py
import torch
import torch.nn as nn
from models.vision_language.internvl2 import InternVL2ReceiptClassifier

class PrototypicalInternVL2(nn.Module):
    """Prototypical network adapting InternVL2 for single-shot learning."""
    
    def __init__(self, config):
        super().__init__()
        # Initialize with pretrained weights
        self.encoder = InternVL2ReceiptClassifier(config, pretrained=True)
        # Remove the original classification head
        del self.encoder.classification_head
        
        # Add prototypical learning components
        self.embedding_dim = 512
        self.embedding_projection = nn.Linear(
            self.encoder.vision_encoder.config.hidden_size, 
            self.embedding_dim
        )
        
    def forward(self, x):
        """Encode input images into the embedding space."""
        outputs = self.encoder.vision_encoder(pixel_values=x)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Project to embedding space
        embedding = self.embedding_projection(pooled_output)
        
        return embedding
        
    def compute_prototypes(self, support_embeddings, support_labels):
        """Compute class prototypes from support embeddings."""
        unique_classes = torch.unique(support_labels)
        prototypes = []
        
        for c in unique_classes:
            # Select embeddings for this class
            mask = support_labels == c
            class_embeddings = support_embeddings[mask]
            
            # Compute mean embedding (prototype)
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
            
        return torch.stack(prototypes)
```

### 2. Loss Function Implementations

#### 2.1. Contrastive Loss for Siamese Networks

```python
# Path: utils/contrastive_loss.py
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for siamese networks.
    
    Brings embeddings of similar pairs closer together, 
    pushes embeddings of dissimilar pairs further apart.
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, emb1, emb2, label):
        """
        Args:
            emb1: Embeddings from first image [batch_size, embedding_dim]
            emb2: Embeddings from second image [batch_size, embedding_dim]
            label: 1 if same class, 0 if different class [batch_size]
            
        Returns:
            Contrastive loss value
        """
        # Compute Euclidean distance between embeddings
        distance = torch.nn.functional.pairwise_distance(emb1, emb2)
        
        # Contrastive loss formula:
        # For similar pairs (label=1): loss = distance²
        # For dissimilar pairs (label=0): loss = max(0, margin - distance)²
        similar_pair_loss = label * torch.pow(distance, 2)
        dissimilar_pair_loss = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        # Mean over batch
        loss = torch.mean(similar_pair_loss + dissimilar_pair_loss) / 2
        
        return loss
```

#### 2.2. Prototypical Loss 

```python
# Path: utils/prototypical_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalLoss(nn.Module):
    """
    Prototypical loss for few-shot learning.
    
    Computes negative log-probability of query samples 
    belonging to their true class prototypes.
    """
    
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query_embeddings, prototypes, query_labels):
        """
        Args:
            query_embeddings: Embeddings of query samples [query_size, embedding_dim]
            prototypes: Class prototypes [n_classes, embedding_dim]
            query_labels: Class labels for query samples [query_size]
            
        Returns:
            Prototypical loss value
        """
        # Compute distance from each query to all prototypes
        distance = torch.cdist(query_embeddings, prototypes)
        
        # Convert distances to probabilities with temperature scaling
        # Negative distance used as logits (smaller distance = higher probability)
        logits = -distance / self.temperature
        
        # Cross-entropy loss with true class labels
        loss = F.cross_entropy(logits, query_labels)
        
        return loss
```

### 3. Data Loading and Preprocessing

#### 3.1. Episode Sampler for Few-Shot Learning

```python
# Path: data/datasets/few_shot/episode_sampler.py
import random
import torch
from torch.utils.data import Sampler

class EpisodeSampler(Sampler):
    """
    Samples episodes (tasks) for few-shot learning.
    
    Each episode contains:
    - support set: n_way * n_shot examples (training)
    - query set: n_way * n_query examples (evaluation)
    """
    
    def __init__(self, dataset, n_way, n_shot, n_query, n_episodes):
        """
        Args:
            dataset: Dataset to sample from
            n_way: Number of classes per episode
            n_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to generate
        """
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        # Group samples by class
        self.samples_by_class = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.samples_by_class:
                self.samples_by_class[label] = []
            self.samples_by_class[label].append(idx)
        
        # Ensure we have enough classes
        assert len(self.samples_by_class) >= n_way, \
            f"Dataset has {len(self.samples_by_class)} classes, but n_way={n_way}"
        
        # Ensure each class has enough samples
        for label, indices in self.samples_by_class.items():
            assert len(indices) >= n_shot + n_query, \
                f"Class {label} has {len(indices)} samples, but needs {n_shot + n_query}"
    
    def __iter__(self):
        """Generate episodes."""
        for _ in range(self.n_episodes):
            # Randomly select n_way classes
            episode_classes = random.sample(list(self.samples_by_class.keys()), self.n_way)
            
            # Sample support and query indices for each class
            support_indices = []
            query_indices = []
            
            for label in episode_classes:
                # Randomly sample indices for this class
                class_indices = random.sample(self.samples_by_class[label], self.n_shot + self.n_query)
                
                # Split into support and query
                support_indices.extend(class_indices[:self.n_shot])
                query_indices.extend(class_indices[self.n_shot:])
            
            # Yield support + query indices for this episode
            yield support_indices + query_indices
    
    def __len__(self):
        """Number of episodes."""
        return self.n_episodes
```

#### 3.2. Episode Dataset for Few-Shot Learning

```python
# Path: data/datasets/few_shot/episode_dataset.py
import torch
from torch.utils.data import Dataset

class EpisodeDataset(Dataset):
    """
    Dataset that provides few-shot learning episodes.
    
    Each episode contains:
    - support set for task adaptation
    - query set for evaluation
    """
    
    def __init__(self, dataset, n_way, n_shot, n_query):
        """
        Args:
            dataset: Base dataset to sample from
            n_way: Number of classes per episode
            n_shot: Number of support examples per class
            n_query: Number of query examples per class
        """
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        
        # Create label-to-index mapping
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # Get list of class labels
        self.labels = list(self.label_to_indices.keys())
        
    def __getitem__(self, index):
        """
        Generate a single episode (task).
        
        Returns:
            Dictionary with:
            - support_images: [n_way * n_shot, C, H, W]
            - support_labels: [n_way * n_shot]
            - query_images: [n_way * n_query, C, H, W]
            - query_labels: [n_way * n_query]
        """
        # Select n_way random classes for this episode
        episode_classes = torch.randperm(len(self.labels))[:self.n_way]
        episode_classes = [self.labels[i] for i in episode_classes]
        
        # Initialize episode data
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        # For each class in this episode
        for class_idx, class_label in enumerate(episode_classes):
            # Get all indices for this class
            class_indices = self.label_to_indices[class_label]
            
            # Sample n_shot + n_query indices
            selected_indices = torch.randperm(len(class_indices))[:self.n_shot + self.n_query]
            
            # Split into support and query sets
            support_indices = selected_indices[:self.n_shot]
            query_indices = selected_indices[self.n_shot:]
            
            # Get support samples
            for idx in support_indices:
                img, _ = self.dataset[class_indices[idx]]
                support_images.append(img)
                # Use relative label within episode (0 to n_way-1)
                support_labels.append(class_idx)
            
            # Get query samples
            for idx in query_indices:
                img, _ = self.dataset[class_indices[idx]]
                query_images.append(img)
                # Use relative label within episode
                query_labels.append(class_idx)
        
        # Convert to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        
        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels
        }
    
    def __len__(self):
        """Number of possible episodes (arbitrary)."""
        return 1000  # This can be adjusted
```

### 4. Training Scripts

#### 4.1. Prototypical Network Training Script

```python
# Path: scripts/training/train_single_shot_prototypical.py
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data.datasets.few_shot.episode_dataset import EpisodeDataset
from data.datasets.classification.balanced_question_dataset import BalancedQuestionDataset
from models.vision_language.prototypical_network import PrototypicalInternVL2
from utils.prototypical_loss import PrototypicalLoss
from utils.device import get_device
from utils.reproducibility import set_seed

def train_prototypical_network(
    config_path: str,
    output_dir: str,
    n_way: int = 3,
    n_shot: int = 1,
    n_query: int = 5,
    num_epochs: int = 30,
    learning_rate: float = 1e-5,
    device = None
):
    """Train prototypical network for single-shot learning."""
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
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup dataset
    train_dataset = BalancedQuestionDataset(split="train")
    val_dataset = BalancedQuestionDataset(split="val")
    
    # Create episode datasets
    train_episodes = EpisodeDataset(train_dataset, n_way, n_shot, n_query)
    val_episodes = EpisodeDataset(val_dataset, n_way, n_shot, n_query)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_episodes,
        batch_size=1,  # Each batch is one episode
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_episodes,
        batch_size=1,  # Each batch is one episode
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = PrototypicalInternVL2(config)
    model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create loss function
    criterion = PrototypicalLoss()
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Extract episode data
            support_images = batch['support_images'].squeeze(0).to(device)
            support_labels = batch['support_labels'].squeeze(0).to(device)
            query_images = batch['query_images'].squeeze(0).to(device)
            query_labels = batch['query_labels'].squeeze(0).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            support_embeddings = model(support_images)
            query_embeddings = model(query_images)
            
            # Compute prototypes
            prototypes = model.compute_prototypes(support_embeddings, support_labels)
            
            # Compute loss
            loss = criterion(query_embeddings, prototypes, query_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute accuracy
            distances = torch.cdist(query_embeddings, prototypes)
            predictions = torch.argmin(distances, dim=1)
            accuracy = (predictions == query_labels).float().mean().item()
            
            # Update metrics
            train_loss += loss.item()
            train_acc += accuracy
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{accuracy:.4f}"
            })
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Extract episode data
                support_images = batch['support_images'].squeeze(0).to(device)
                support_labels = batch['support_labels'].squeeze(0).to(device)
                query_images = batch['query_images'].squeeze(0).to(device)
                query_labels = batch['query_labels'].squeeze(0).to(device)
                
                # Forward pass
                support_embeddings = model(support_images)
                query_embeddings = model(query_images)
                
                # Compute prototypes
                prototypes = model.compute_prototypes(support_embeddings, support_labels)
                
                # Compute loss
                loss = criterion(query_embeddings, prototypes, query_labels)
                
                # Compute accuracy
                distances = torch.cdist(query_embeddings, prototypes)
                predictions = torch.argmin(distances, dim=1)
                accuracy = (predictions == query_labels).float().mean().item()
                
                # Update metrics
                val_loss += loss.item()
                val_acc += accuracy
        
        # Calculate average metrics
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            
            # Save model
            save_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'n_way': n_way,
                    'n_shot': n_shot
                }
            }, save_path)
            logger.info(f"Model saved to {save_path}")
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'best_val_acc': best_val_acc
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train prototypical network for single-shot learning")
    parser.add_argument("--config", type=str, default="config/model/multimodal_config.yaml", help="Path to model config")
    parser.add_argument("--output-dir", type=str, default="models/single_shot", help="Output directory")
    parser.add_argument("--n-way", type=int, default=3, help="Number of classes per episode")
    parser.add_argument("--n-shot", type=int, default=1, help="Number of support examples per class")
    parser.add_argument("--n-query", type=int, default=5, help="Number of query examples per class")
    parser.add_argument("--num-epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Train model
    train_prototypical_network(
        config_path=args.config,
        output_dir=args.output_dir,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()
```

### 5. Inference and Adaptation Scripts

#### 5.1. Inference with Prototypical Network

```python
# Path: scripts/inference/single_shot_inference.py
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from models.vision_language.prototypical_network import PrototypicalInternVL2
from utils.device import get_device

def load_model(model_path: str, config_path: str, device) -> PrototypicalInternVL2:
    """Load pretrained prototypical network model."""
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = PrototypicalInternVL2(config)
    
    # Load saved state
    saved_state = torch.load(model_path, map_location=device)
    model.load_state_dict(saved_state['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    return model

def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess an image for model input."""
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

def single_shot_inference(
    model: PrototypicalInternVL2,
    support_images: List[str],
    support_labels: List[int],
    query_image: str,
    device
) -> Tuple[int, Dict[int, float]]:
    """
    Perform single-shot inference with a prototypical network.
    
    Args:
        model: Pretrained prototypical network
        support_images: List of paths to support images
        support_labels: Class labels for support images
        query_image: Path to query image
        device: Device to run inference on
        
    Returns:
        Predicted class and confidence scores
    """
    # Preprocess images
    support_tensors = []
    for img_path in support_images:
        img_tensor = preprocess_image(img_path)
        support_tensors.append(img_tensor)
    
    support_tensors = torch.cat(support_tensors, dim=0).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    query_tensor = preprocess_image(query_image).to(device)
    
    # Get embeddings
    with torch.no_grad():
        support_embeddings = model(support_tensors)
        query_embedding = model(query_tensor)
        
        # Compute prototypes
        prototypes = model.compute_prototypes(support_embeddings, support_labels)
        
        # Compute distances to prototypes
        distances = torch.cdist(query_embedding, prototypes)[0]
        
        # Convert distances to probabilities (using softmax with negative distances)
        probs = F.softmax(-distances, dim=0)
        
        # Get prediction
        pred_class = torch.argmin(distances).item()
        
        # Create confidence scores dictionary
        confidence_scores = {i.item(): prob.item() for i, prob in enumerate(probs)}
    
    return pred_class, confidence_scores

def main():
    """Main function for single-shot inference."""
    parser = argparse.ArgumentParser(description="Single-shot inference with prototypical network")
    parser.add_argument("--model", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--support-dir", type=str, required=True, help="Directory containing support images")
    parser.add_argument("--query-image", type=str, required=True, help="Path to query image")
    parser.add_argument("--labels-file", type=str, required=True, help="JSON file mapping support images to labels")
    
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
    model = load_model(args.model, args.config, device)
    
    # Load support image labels
    import json
    with open(args.labels_file, 'r') as f:
        labels_data = json.load(f)
    
    # Prepare support data
    support_images = []
    support_labels = []
    
    for img_name, label in labels_data.items():
        img_path = os.path.join(args.support_dir, img_name)
        if os.path.exists(img_path):
            support_images.append(img_path)
            support_labels.append(label)
    
    # Perform inference
    logger.info(f"Performing single-shot inference with {len(support_images)} support images")
    pred_class, confidences = single_shot_inference(
        model, support_images, support_labels, args.query_image, device
    )
    
    # Log results
    logger.info(f"Predicted class: {pred_class}")
    logger.info("Confidence scores:")
    for class_idx, confidence in confidences.items():
        logger.info(f"  Class {class_idx}: {confidence:.4f}")
    
    # Format for output file
    result = {
        "predicted_class": pred_class,
        "confidence_scores": confidences
    }
    
    # Save results
    output_path = os.path.join(os.path.dirname(args.query_image), "results.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
```

### 6. Configuration Updates

#### 6.1. Model Configuration

Create a new configuration file for single-shot learning:

```yaml
# Path: config/model/single_shot_config.yaml
model:
  # Base model configuration
  pretrained_path: "/Users/tod/PretrainedLLM/InternVL2_5-1B"
  multimodal: true
  
  # Single-shot learning specific configuration
  single_shot:
    enabled: true
    architecture: "prototypical"  # Options: "prototypical", "siamese"
    embedding_dim: 512
    use_pretrained: true
    temperature: 0.5  # Temperature for softmax in prototypical networks
    freeze_base_model: true  # Whether to freeze the base InternVL2 model
  
  # Classifier components (used only if not in single-shot mode)
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
  # Single-shot training configuration
  single_shot:
    n_way: 3
    n_shot: 1
    n_query: 5
    episodes_per_epoch: 100
    evaluation_episodes: 50
  
  # Standard training configuration (used as fallback if not in single-shot mode)
  epochs: 30
  learning_rate: 1.0e-5
  gradient_clip: 1.0
  weight_decay: 0.0001
  optimizer:
    name: "adamw"
    learning_rate: 2.0e-5
    weight_decay: 0.01
  
  early_stopping:
    patience: 10
    min_delta: 0.001
  
  scheduler:
    name: "cosine"
    warmup_steps: 500
```

### 7. Troubleshooting and Debugging Guide

#### Common Issues and Solutions

1. **Out of Memory (OOM) Errors**
   - Reduce batch size or n_query parameter
   - Enable gradient checkpointing by adding the flag `--gradient-checkpointing` to training scripts
   - Use half-precision training with `--fp16` flag

2. **Low Accuracy with Few Examples**
   - Increase embedding dimension in `single_shot_config.yaml`
   - Add data augmentation to support samples
   - Try using the siamese architecture instead of prototypical for better performance on small datasets

3. **Slow Training**
   - Ensure you're using GPU acceleration
   - Reduce number of episodes per epoch
   - Use a smaller base model

4. **Overfitting on Support Set**
   - Increase regularization by adding dropout to embedding projector
   - Use data augmentation on support set
   - Add L2 regularization to model parameters

### 8. Testing Migration

To ensure your migration was successful:

1. Create a small test dataset with just 1-3 examples per class
2. Run the single-shot training script:
   ```
   PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/internvl_env/bin/python scripts/training/train_single_shot_prototypical.py --config config/model/single_shot_config.yaml --n-shot 1 --n-way 3
   ```
3. Verify if model achieves reasonable accuracy (>70%) with just one example per class
4. Test inference with new examples not seen during training

### 9. Considerations for Production

1. **Model Versioning**
   - Implement version tracking for single-shot models
   - Store support sets along with model checkpoints for reproducibility

2. **Serving Efficiency**
   - Cache embeddings for support set to avoid recomputing them
   - Implement batched inference for multiple query images

3. **Continual Learning**
   - Add ability to update support set without retraining
   - Implement memory to store challenging examples for future training

4. **Fallback Mechanisms**
   - Configure system to fall back to traditional model when confidence is low
   - Implement confidence thresholds for requesting human verification

### 10. Potential Extensions

1. **Cross-modal Single-Shot Learning**
   - Extend to learn from text descriptions with one image example
   - Implement multimodal prototypes combining visual and text features

2. **Meta-Learning Improvements**
   - Add Model-Agnostic Meta-Learning (MAML) implementation
   - Implement Reptile algorithm for faster adaptation

3. **Dynamic Support Set Management**
   - Add automatic cleaning of support set to remove outliers
   - Implement smart example selection to maximize coverage of class variation

4. **Zero-Shot Transfer**
   - Extend to recognize completely new classes from text descriptions only
   - Add class name semantic embedding to enhance zero-shot capabilities

## Conclusion

This migration guide provides a comprehensive approach to adapting InternVL_V2 for single-shot learning. Following these steps will allow the model to recognize new receipt types with just one or a few examples, significantly reducing data requirements and enabling faster adaptation to new document formats.

By implementing either the siamese or prototypical network approaches, you'll enhance the model's ability to generalize from minimal examples while maintaining high accuracy - a critical capability for document processing systems that need to handle variable formats with minimal training data.