"""
Dataset implementations for receipt counting and multimodal vision-language tasks.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class ReceiptDataset(Dataset):
    """
    Dataset for receipt counting with InternVL2 support.
    
    Implements efficient loading and preprocessing with torchvision transforms.
    """
    def __init__(
        self,
        csv_file: Union[str, Path],
        img_dir: Union[str, Path],
        transform: Optional[T.Compose] = None,
        augment: bool = False,
        binary: bool = False,
        max_samples: Optional[int] = None,
        image_size: int = 448,
    ):
        """
        Initialize a receipt dataset.
        
        Args:
            csv_file: Path to CSV file containing image filenames and receipt counts
            img_dir: Directory containing the images
            transform: Optional custom transform to apply to images
            image_size: Target image size (448 for InternVL2)
            augment: Whether to apply data augmentation (used for training)
            binary: Whether to use binary classification mode (0 vs 1+ receipts)
            max_samples: Optional limit on number of samples (useful for quick testing)
        """
        self.data = pd.read_csv(csv_file)
        if max_samples is not None:
            self.data = self.data.sample(min(len(self.data), max_samples))
            
        self.img_dir = Path(img_dir)
        self.binary = binary
        self.image_size = image_size
        
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
                    T.RandomVerticalFlip(p=0.1),  # Occasionally flip vertically (some receipts are upside down)
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.data.iloc[idx, 0]
        image_path = self.img_dir / filename
        
        try:
            # Read image using PIL
            image = Image.open(image_path).convert('RGB')
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback to a blank image
            image = Image.new('RGB', (self.image_size, self.image_size), color=(0, 0, 0))
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Get receipt count
        count = int(self.data.iloc[idx, 1])
        
        # Convert to appropriate classification label
        if self.binary:
            # Binary: 0 vs 1+ receipts
            label = 1 if count > 0 else 0
        else:
            # Default: 3-class (0, 1, 2+ receipts)
            label = min(count, 2)  # Map all counts ≥2 to class 2
            
        return image_tensor, torch.tensor(label, dtype=torch.long)


class MultimodalReceiptDataset(Dataset):
    """
    Multimodal dataset for vision-language tasks with receipts.
    
    Handles both images and text for question-answering about receipts.
    """
    def __init__(
        self,
        csv_file: Union[str, Path],
        img_dir: Union[str, Path],
        tokenizer_path: Union[str, Path],
        transform: Optional[T.Compose] = None, 
        max_length: int = 128,
        augment: bool = False,
        max_samples: Optional[int] = None,
        image_size: int = 448,
    ):
        """
        Initialize a multimodal receipt dataset.
        
        Args:
            csv_file: Path to CSV file with image filenames, receipt counts, questions, and answers
            img_dir: Directory containing the images
            tokenizer_path: Path to pretrained tokenizer
            transform: Optional custom transform for images
            max_length: Maximum length for text sequences
            augment: Whether to apply data augmentation (used for training)
            max_samples: Optional limit on number of samples (useful for quick testing)
            image_size: Target image size (448 for InternVL2)
        """
        self.data = pd.read_csv(csv_file)
        if max_samples is not None:
            self.data = self.data.sample(min(len(self.data), max_samples))
            
        self.img_dir = Path(img_dir)
        self.image_size = image_size
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        
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
        filename = self.data.iloc[idx]["filename"]
        image_path = self.img_dir / filename
        receipt_count = int(self.data.iloc[idx]["receipt_count"])
        question = str(self.data.iloc[idx]["question"])
        answer = str(self.data.iloc[idx]["answer"])
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Fallback to a blank image
            image = Image.new('RGB', (self.image_size, self.image_size), color=(0, 0, 0))
        
        # Apply transformations
        image_tensor = self.transform(image)
        
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
        
        # Create label for classification (for multi-task learning)
        # Default: 3-class (0, 1, 2+ receipts)
        classification_label = min(receipt_count, 2)  # Map all counts ≥2 to class 2
        
        return {
            "pixel_values": image_tensor,
            "text_input_ids": question_input_ids,
            "text_attention_mask": question_attention_mask,
            "labels": answer_input_ids,
            "labels_attention_mask": answer_attention_mask,
            "classification_labels": torch.tensor(classification_label, dtype=torch.long),
            "receipt_count": torch.tensor(receipt_count, dtype=torch.long),
        }


def create_dataloaders(config) -> Dict[str, DataLoader]:
    """
    Create training, validation, and test data loaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing train_loader, val_loader, test_loader
    """
    # Check if we're using multimodal mode
    is_multimodal = config.get("model", {}).get("multimodal", False)
    
    if is_multimodal:
        # Create multimodal datasets
        train_dataset = MultimodalReceiptDataset(
            config["data"]["train_csv"],
            config["data"]["train_dir"],
            config["model"]["pretrained_path"],  # Using same path for tokenizer
            augment=config["data"]["augmentation"],
            max_length=config["data"].get("max_text_length", 128),
            image_size=config["data"]["image_size"],
        )
        
        val_dataset = MultimodalReceiptDataset(
            config["data"]["val_csv"],
            config["data"]["val_dir"],
            config["model"]["pretrained_path"],  # Using same path for tokenizer
            augment=False,
            max_length=config["data"].get("max_text_length", 128),
            image_size=config["data"]["image_size"],
        )
        
        test_dataset = None
        if "test_csv" in config["data"] and "test_dir" in config["data"]:
            test_dataset = MultimodalReceiptDataset(
                config["data"]["test_csv"],
                config["data"]["test_dir"],
                config["model"]["pretrained_path"],  # Using same path for tokenizer
                augment=False,
                max_length=config["data"].get("max_text_length", 128),
                image_size=config["data"]["image_size"],
            )
    else:
        # Create vision-only datasets
        train_dataset = ReceiptDataset(
            config["data"]["train_csv"],
            config["data"]["train_dir"],
            augment=config["data"]["augmentation"],
            binary=config["model"]["num_classes"] == 2,
            image_size=config["data"]["image_size"],
        )
        
        val_dataset = ReceiptDataset(
            config["data"]["val_csv"],
            config["data"]["val_dir"],
            augment=False,
            binary=config["model"]["num_classes"] == 2,
            image_size=config["data"]["image_size"],
        )
        
        test_dataset = None
        if "test_csv" in config["data"] and "test_dir" in config["data"]:
            test_dataset = ReceiptDataset(
                config["data"]["test_csv"],
                config["data"]["test_dir"],
                augment=False,
                binary=config["model"]["num_classes"] == 2,
                image_size=config["data"]["image_size"],
            )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
    }
    
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True,
        )
        loaders['test'] = test_loader
    
    return loaders


def collate_fn_multimodal(batch):
    """
    Custom collate function for multimodal batching.
    
    Ensures proper handling of variable-length text sequences.
    
    Args:
        batch: List of samples
        
    Returns:
        Collated batch with tensors
    """
    # Extract elements
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    text_input_ids = torch.stack([item["text_input_ids"] for item in batch])
    text_attention_mask = torch.stack([item["text_attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    labels_attention_mask = torch.stack([item["labels_attention_mask"] for item in batch])
    classification_labels = torch.stack([item["classification_labels"] for item in batch])
    receipt_counts = torch.stack([item["receipt_count"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "text_input_ids": text_input_ids,
        "text_attention_mask": text_attention_mask,
        "labels": labels,
        "labels_attention_mask": labels_attention_mask,
        "classification_labels": classification_labels,
        "receipt_count": receipt_counts,
    }