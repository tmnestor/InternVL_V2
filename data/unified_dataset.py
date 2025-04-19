"""
Unified dataset implementation for both receipt counting and multimodal vision-language tasks.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class UnifiedDataset(Dataset):
    """
    Unified dataset for both receipt counting and multimodal QA tasks.
    
    Handles both basic receipt counting and question answering in a single dataset.
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        tokenizer_path: Optional[Union[str, Path]] = None,
        transform: Optional[T.Compose] = None, 
        max_length: int = 128,
        augment: bool = False,
        max_samples: Optional[int] = None,
        image_size: int = 448,
    ):
        """
        Initialize a unified dataset for both counting and QA tasks.
        
        Args:
            data_dir: Path to the dataset directory
            split: Split to use (train, val, test)
            tokenizer_path: Path to pretrained tokenizer (required for QA)
            transform: Optional custom transform for images
            max_length: Maximum length for text sequences
            augment: Whether to apply data augmentation (used for training)
            max_samples: Optional limit on number of samples
            image_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_length = max_length
        
        # Load metadata
        metadata_path = self.data_dir / split / "metadata.csv"
        self.data = pd.read_csv(metadata_path)
        
        if max_samples is not None:
            self.data = self.data.sample(min(len(self.data), max_samples))
            
        # Set image directory
        self.img_dir = self.data_dir / split / "images"
        
        # Load tokenizer if path is provided (required for QA)
        self.tokenizer = None
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                local_files_only=True, 
                trust_remote_code=True
            )
        
        # Set up transforms
        if transform:
            self.transform = transform
        else:
            # ImageNet normalization values
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            # Default transformations based on model requirements
            if augment:
                self.transform = T.Compose([
                    T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.1),  # Occasionally flip vertically
                    T.RandomAffine(
                        degrees=30,
                        translate=(0.1, 0.1),
                        scale=(0.8, 1.2),
                    ),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    normalize,
                ])
            else:
                # Validation/test transforms
                self.transform = T.Compose([
                    T.Resize((self.image_size, self.image_size)),
                    T.ToTensor(),
                    normalize,
                ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get data for this sample
        row = self.data.iloc[idx]
        filename = row["filename"]
        image_path = self.img_dir / filename
        receipt_count = int(row["receipt_count"])
        
        try:
            # Read image using PIL
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback to a blank image
            image = Image.new('RGB', (self.image_size, self.image_size), color=(0, 0, 0))
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Create label for classification (3-class: 0, 1, 2+ receipts)
        classification_label = min(receipt_count, 2)  # Map all counts ≥2 to class 2
        
        # Base output with image and classification label
        output = {
            "pixel_values": image_tensor,
            "classification_labels": torch.tensor(classification_label, dtype=torch.long),
            "receipt_count": torch.tensor(receipt_count, dtype=torch.long),
        }
        
        # Add QA data if available in the dataset
        if "question" in row and "answer" in row and self.tokenizer is not None:
            question = str(row["question"])
            answer = str(row["answer"])
            
            # Tokenize text
            question_encoding = self.tokenizer(
                question,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            answer_encoding = self.tokenizer(
                answer,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Remove batch dimension from tokenizer output
            question_input_ids = question_encoding.input_ids.squeeze(0)
            question_attention_mask = question_encoding.attention_mask.squeeze(0)
            answer_input_ids = answer_encoding.input_ids.squeeze(0)
            answer_attention_mask = answer_encoding.attention_mask.squeeze(0)
            
            # Add QA fields to the output
            output.update({
                "text_input_ids": question_input_ids,
                "text_attention_mask": question_attention_mask,
                "labels": answer_input_ids,
                "labels_attention_mask": answer_attention_mask,
                "original_question": question,
                "answer": answer,
            })
            
            # Determine question type if available
            if "question_type" in row:
                output["question_type"] = row["question_type"]
            
        return output


def create_unified_dataloader(
    config: Dict,
    split: str = "train",
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader for the unified dataset.
    
    Args:
        config: Configuration dictionary
        split: Split to use (train, val, test)
        
    Returns:
        Configured DataLoader for the unified dataset
    """
    # Check if multimodal mode is enabled (for QA)
    is_multimodal = config.get("model", {}).get("multimodal", False)
    
    # Setup tokenizer path if multimodal
    tokenizer_path = None
    if is_multimodal:
        tokenizer_path = config["model"]["pretrained_path"]
    
    # Create unified dataset
    dataset = UnifiedDataset(
        data_dir=config["data"]["root_dir"],
        split=split,
        tokenizer_path=tokenizer_path,
        augment=(split == "train") and config["data"]["augmentation"],
        max_length=config["data"].get("max_text_length", 128),
        image_size=config["data"]["image_size"],
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=(split == "train"),
    )
    
    return dataloader