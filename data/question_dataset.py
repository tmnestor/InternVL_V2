"""
Dataset for training question classifiers and template-based responses.

This module provides datasets and data generators for training question classification
and response generation components of the multimodal system.
"""
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset
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
        tokenizer_name: str = "ModernBert-base",
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
        
        # Check for custom ModernBert path
        logger = logging.getLogger(__name__)
        custom_path = "/home/jovyan/nfs_share/models/huggingface/hub/ModernBERT-base"
        
        # Try to initialize tokenizer with detailed logging
        try:
            # First check if we should use the custom path
            if os.path.exists(custom_path) and (tokenizer_name == "ModernBert-base" or tokenizer_name == custom_path):
                logger.info(f"TOKENIZER CHECK: Loading tokenizer from custom path: {custom_path}")
                
                # Verify path and contents
                if os.path.exists(custom_path):
                    logger.info(f"TOKENIZER CHECK: Path exists. Contents: {os.listdir(custom_path)}")
                    
                    # Check for tokenizer files
                    tokenizer_files = [f for f in os.listdir(custom_path) if "tokenizer" in f.lower()]
                    if tokenizer_files:
                        logger.info(f"TOKENIZER CHECK: Found tokenizer files: {tokenizer_files}")
                    else:
                        logger.warning("TOKENIZER CHECK: No tokenizer files found")
                    
                    # Check for vocab files
                    vocab_files = [f for f in os.listdir(custom_path) if "vocab" in f.lower()]
                    if vocab_files:
                        logger.info(f"TOKENIZER CHECK: Found vocab files: {vocab_files}")
                    else:
                        logger.warning("TOKENIZER CHECK: No vocab files found")
                else:
                    logger.warning(f"TOKENIZER CHECK: Path does not exist: {custom_path}")
                
                # Attempt to load tokenizer
                logger.info("TOKENIZER CHECK: Calling AutoTokenizer.from_pretrained with local_files_only=True")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    custom_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                logger.info(f"TOKENIZER CHECK: Successfully loaded tokenizer. Type: {type(self.tokenizer).__name__}")
                logger.info(f"TOKENIZER CHECK: Vocab size: {len(self.tokenizer.get_vocab())}")
                logger.info(f"TOKENIZER CHECK: Sample token IDs for 'hello': {self.tokenizer.encode('hello')}")
            else:
                # Use the provided tokenizer name
                logger.info(f"TOKENIZER CHECK: Loading tokenizer from provided name: {tokenizer_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                logger.info(f"TOKENIZER CHECK: Successfully loaded tokenizer from hub. Type: {type(self.tokenizer).__name__}")
        except Exception as e:
            logger.warning(f"TOKENIZER CHECK: Error loading tokenizer: {e}")
            logger.warning(f"TOKENIZER CHECK: Exception type: {type(e).__name__}")
            logger.warning(f"TOKENIZER CHECK: Exception details: {str(e)}")
            logger.info("TOKENIZER CHECK: Falling back to default tokenizer distilbert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            logger.info(f"TOKENIZER CHECK: Fallback tokenizer loaded. Type: {type(self.tokenizer).__name__}")
        
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
    tokenizer_name: str = "ModernBert-base",
    max_length: int = 128,
    num_workers: int = 0,
    use_custom_path: bool = True
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for question classification training.
    
    Args:
        data_dir: Directory containing question data
        batch_size: Batch size for dataloaders
        tokenizer_name: Name of tokenizer to use
        max_length: Maximum sequence length
        num_workers: Number of workers for dataloaders
        use_custom_path: Whether to set the cache environment variable
    """
    # Set the transformers cache if requested and path exists
    custom_cache_dir = "/home/jovyan/nfs_share/models/huggingface/hub"
    if use_custom_path and not os.environ.get("TRANSFORMERS_CACHE") and os.path.exists(custom_cache_dir):
        os.environ["TRANSFORMERS_CACHE"] = custom_cache_dir
        logging.info(f"Set TRANSFORMERS_CACHE to {custom_cache_dir}")
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