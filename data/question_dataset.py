"""
Dataset for training question classifiers and template-based responses.

This module provides datasets and data generators for training question classification
and response generation components of the multimodal system.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class QuestionDataset(Dataset):
    """
    Dataset for training question classification models.
    
    Contains labeled questions for different types of document queries
    to train the question classifier component.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        split: str = "train",
        create_default: bool = True,
    ):
        """
        Initialize question dataset.
        
        Args:
            data_dir: Directory containing question data
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length for tokenization
            split: Dataset split (train, val, test)
            create_default: Whether to create default dataset if not found
        """
        # Convert to Path object
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.split = split
        
        # Define file path for dataset
        self.file_path = self.data_dir / f"question_dataset_{split}.json"
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load or create dataset
        if self.file_path.exists():
            self.questions = self._load_dataset()
        elif create_default:
            self.questions = self._create_default_dataset()
            self._save_dataset()
        else:
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Define question type mapping
        self.question_types = {
            "DOCUMENT_TYPE": 0,
            "COUNTING": 1,
            "DETAIL_EXTRACTION": 2,
            "PAYMENT_INFO": 3,
            "TAX_INFO": 4,
        }
    
    def _load_dataset(self) -> List[Dict]:
        """
        Load dataset from file.
        
        Returns:
            List of question dictionaries
        """
        with open(self.file_path, "r") as f:
            return json.load(f)
    
    def _save_dataset(self):
        """Save dataset to file."""
        with open(self.file_path, "w") as f:
            json.dump(self.questions, f, indent=2)
    
    def _create_default_dataset(self) -> List[Dict]:
        """
        Create default question dataset.
        
        Returns:
            List of question dictionaries
        """
        # Basic set of questions for each type
        questions = [
            # Document type questions
            {"question": "Is this a receipt?", "type": "DOCUMENT_TYPE"},
            {"question": "What kind of document is this?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this a tax document?", "type": "DOCUMENT_TYPE"},
            {"question": "What type of document am I looking at?", "type": "DOCUMENT_TYPE"},
            {"question": "Can you identify this document?", "type": "DOCUMENT_TYPE"},
            
            # Counting questions
            {"question": "How many receipts are in this image?", "type": "COUNTING"},
            {"question": "Count the number of receipts.", "type": "COUNTING"},
            {"question": "Are there multiple receipts here?", "type": "COUNTING"},
            {"question": "How many receipts do you see?", "type": "COUNTING"},
            {"question": "Can you count the receipts in this image?", "type": "COUNTING"},
            
            # Detail extraction questions
            {"question": "What store is this receipt from?", "type": "DETAIL_EXTRACTION"},
            {"question": "What is the date on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What items were purchased?", "type": "DETAIL_EXTRACTION"},
            {"question": "When was this purchase made?", "type": "DETAIL_EXTRACTION"},
            {"question": "What store issued this receipt?", "type": "DETAIL_EXTRACTION"},
            
            # Payment information questions
            {"question": "How was this purchase paid for?", "type": "PAYMENT_INFO"},
            {"question": "What payment method was used?", "type": "PAYMENT_INFO"},
            {"question": "Was this paid by credit card?", "type": "PAYMENT_INFO"},
            {"question": "What was the payment type?", "type": "PAYMENT_INFO"},
            {"question": "Did they pay with cash or card?", "type": "PAYMENT_INFO"},
            
            # Tax document questions
            {"question": "What tax form is this?", "type": "TAX_INFO"},
            {"question": "What is the ABN on this document?", "type": "TAX_INFO"},
            {"question": "What tax year does this document cover?", "type": "TAX_INFO"},
            {"question": "Is this an official ATO document?", "type": "TAX_INFO"},
            {"question": "What is the tax file number?", "type": "TAX_INFO"},
        ]
        
        return questions
    
    def expand_dataset(self, additional_questions: List[Dict]):
        """
        Expand dataset with additional questions.
        
        Args:
            additional_questions: List of additional question dictionaries
        """
        self.questions.extend(additional_questions)
        self._save_dataset()
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get item by index.
        
        Args:
            idx: Index to retrieve
            
        Returns:
            Dictionary with tokenized question and label
        """
        item = self.questions[idx]
        question = item["question"]
        question_type = item["type"]
        
        # Tokenize question
        inputs = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Get label index
        label = self.question_types[question_type]
        
        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "question": question,
            "question_type": question_type
        }


def create_question_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for question classification training.
    
    Args:
        data_dir: Directory containing question data
        batch_size: Batch size for dataloaders
        tokenizer_name: Name of tokenizer to use
        max_length: Maximum sequence length
        num_workers: Number of workers for dataloaders
        
    Returns:
        Dictionary of dataloaders for train, val, test splits
    """
    # Create datasets
    train_dataset = QuestionDataset(
        data_dir=data_dir,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="train",
        create_default=True
    )
    
    val_dataset = QuestionDataset(
        data_dir=data_dir,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="val",
        create_default=True
    )
    
    test_dataset = QuestionDataset(
        data_dir=data_dir,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="test",
        create_default=True
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }