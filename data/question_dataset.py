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
        Create expanded default question dataset with many more examples.
        
        Returns:
            List of question dictionaries
        """
        # Much larger and more diverse set of questions for each type
        questions = [
            # Document type questions - 20 examples
            {"question": "Is this a receipt?", "type": "DOCUMENT_TYPE"},
            {"question": "What kind of document is this?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this a tax document?", "type": "DOCUMENT_TYPE"},
            {"question": "What type of document am I looking at?", "type": "DOCUMENT_TYPE"},
            {"question": "Can you identify this document?", "type": "DOCUMENT_TYPE"},
            {"question": "What document do I have here?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this a receipt or a tax document?", "type": "DOCUMENT_TYPE"},
            {"question": "Tell me what kind of document this is.", "type": "DOCUMENT_TYPE"},
            {"question": "What am I looking at in this image?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this a receipt from a store?", "type": "DOCUMENT_TYPE"},
            {"question": "Does this look like a receipt to you?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this document from the ATO?", "type": "DOCUMENT_TYPE"},
            {"question": "What would you call this type of document?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this an official document or a receipt?", "type": "DOCUMENT_TYPE"},
            {"question": "What category of document is shown?", "type": "DOCUMENT_TYPE"},
            {"question": "Can you tell if this is a receipt?", "type": "DOCUMENT_TYPE"},
            {"question": "Would you classify this as a tax document?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this a receipt or something else?", "type": "DOCUMENT_TYPE"},
            {"question": "What do you call this kind of document?", "type": "DOCUMENT_TYPE"},
            {"question": "Can you identify whether this is a receipt?", "type": "DOCUMENT_TYPE"},
            
            # Counting questions - 20 examples
            {"question": "How many receipts are in this image?", "type": "COUNTING"},
            {"question": "Count the number of receipts.", "type": "COUNTING"},
            {"question": "Are there multiple receipts here?", "type": "COUNTING"},
            {"question": "How many receipts do you see?", "type": "COUNTING"},
            {"question": "Can you count the receipts in this image?", "type": "COUNTING"},
            {"question": "Tell me how many receipts are shown.", "type": "COUNTING"},
            {"question": "What's the count of receipts in this picture?", "type": "COUNTING"},
            {"question": "Is there more than one receipt?", "type": "COUNTING"},
            {"question": "How many separate receipts can you identify?", "type": "COUNTING"},
            {"question": "Count the receipts for me please.", "type": "COUNTING"},
            {"question": "Give me the number of receipts visible.", "type": "COUNTING"},
            {"question": "Is there just one receipt or multiple?", "type": "COUNTING"},
            {"question": "Can you enumerate the receipts shown?", "type": "COUNTING"},
            {"question": "What's the total number of receipts?", "type": "COUNTING"},
            {"question": "Tell me the receipt count in this image.", "type": "COUNTING"},
            {"question": "How many distinct receipts are there?", "type": "COUNTING"},
            {"question": "Count how many receipts are present.", "type": "COUNTING"},
            {"question": "Are there two or more receipts shown?", "type": "COUNTING"},
            {"question": "What is the quantity of receipts in this image?", "type": "COUNTING"},
            {"question": "How many individual receipts can you find?", "type": "COUNTING"},
            
            # Detail extraction questions - 40+ examples (doubled with more variation)
            {"question": "What store is this receipt from?", "type": "DETAIL_EXTRACTION"},
            {"question": "What is the date on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What items were purchased?", "type": "DETAIL_EXTRACTION"},
            {"question": "When was this purchase made?", "type": "DETAIL_EXTRACTION"},
            {"question": "What store issued this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What was bought according to this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What time was this purchase made?", "type": "DETAIL_EXTRACTION"},
            {"question": "Which products are listed on the receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What is the name of the store on the receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "Can you tell me the purchase date?", "type": "DETAIL_EXTRACTION"},
            {"question": "What items did they buy?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the store name on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "When was this transaction completed?", "type": "DETAIL_EXTRACTION"},
            {"question": "What products were purchased?", "type": "DETAIL_EXTRACTION"},
            {"question": "Which retailer issued this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What date is shown on the receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the transaction date?", "type": "DETAIL_EXTRACTION"},
            {"question": "Tell me what was purchased.", "type": "DETAIL_EXTRACTION"},
            {"question": "What does the receipt say was bought?", "type": "DETAIL_EXTRACTION"},
            {"question": "What merchant issued this receipt?", "type": "DETAIL_EXTRACTION"},
            
            # Additional detail extraction examples - focus on specific receipt details
            {"question": "What's the GST amount on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the subtotal before tax?", "type": "DETAIL_EXTRACTION"},
            {"question": "How much tax was charged?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the ABN listed on the receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the receipt number?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the transaction ID on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the cashier's name on the receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What register number was used?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the store's address on the receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the store's phone number?", "type": "DETAIL_EXTRACTION"},
            {"question": "What time was the purchase made?", "type": "DETAIL_EXTRACTION"},
            {"question": "How many items were purchased?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the price of the first item?", "type": "DETAIL_EXTRACTION"},
            {"question": "What discount was applied?", "type": "DETAIL_EXTRACTION"},
            {"question": "Was there a loyalty discount applied?", "type": "DETAIL_EXTRACTION"},
            {"question": "What department was this purchased from?", "type": "DETAIL_EXTRACTION"},
            {"question": "What brand is shown on the receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the quantity of each item purchased?", "type": "DETAIL_EXTRACTION"},
            {"question": "What was the most expensive item purchased?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the store's website listed on the receipt?", "type": "DETAIL_EXTRACTION"},
            
            # Additional receipt information variations
            {"question": "Extract the total amount from this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "What are the line items on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "List all products purchased according to the receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "What are the individual prices of items on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "Tell me the date and time of this transaction.", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the business name that issued this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What are the quantities of each item purchased?", "type": "DETAIL_EXTRACTION"},
            {"question": "Read the receipt and tell me what was bought.", "type": "DETAIL_EXTRACTION"},
            {"question": "Read the receipt and tell me when it was issued.", "type": "DETAIL_EXTRACTION"},
            {"question": "What is the exact purchase time on this receipt?", "type": "DETAIL_EXTRACTION"},
            
            # Payment information questions - 20 examples
            {"question": "How was this purchase paid for?", "type": "PAYMENT_INFO"},
            {"question": "What payment method was used?", "type": "PAYMENT_INFO"},
            {"question": "Was this paid by credit card?", "type": "PAYMENT_INFO"},
            {"question": "What was the payment type?", "type": "PAYMENT_INFO"},
            {"question": "Did they pay with cash or card?", "type": "PAYMENT_INFO"},
            {"question": "What form of payment was used?", "type": "PAYMENT_INFO"},
            {"question": "How did the customer pay?", "type": "PAYMENT_INFO"},
            {"question": "Was this paid in cash?", "type": "PAYMENT_INFO"},
            {"question": "Which payment option was selected?", "type": "PAYMENT_INFO"},
            {"question": "What payment card was used?", "type": "PAYMENT_INFO"},
            {"question": "Was this purchase made with EFTPOS?", "type": "PAYMENT_INFO"},
            {"question": "What payment details are shown?", "type": "PAYMENT_INFO"},
            {"question": "Can you tell how they paid?", "type": "PAYMENT_INFO"},
            {"question": "What does it say about the payment method?", "type": "PAYMENT_INFO"},
            {"question": "Does it show how payment was made?", "type": "PAYMENT_INFO"},
            {"question": "Tell me about the payment method used.", "type": "PAYMENT_INFO"},
            {"question": "How was the transaction settled?", "type": "PAYMENT_INFO"},
            {"question": "What payment information is on the receipt?", "type": "PAYMENT_INFO"},
            {"question": "Did they use a debit card?", "type": "PAYMENT_INFO"},
            {"question": "Can you identify the payment method?", "type": "PAYMENT_INFO"},
            
            # Tax document questions - 20 examples
            {"question": "What tax form is this?", "type": "TAX_INFO"},
            {"question": "What is the ABN on this document?", "type": "TAX_INFO"},
            {"question": "What tax year does this document cover?", "type": "TAX_INFO"},
            {"question": "Is this an official ATO document?", "type": "TAX_INFO"},
            {"question": "What is the tax file number?", "type": "TAX_INFO"},
            {"question": "Which financial year is this tax document for?", "type": "TAX_INFO"},
            {"question": "Can you find the ABN listed?", "type": "TAX_INFO"},
            {"question": "What's the TFN shown on this document?", "type": "TAX_INFO"},
            {"question": "Is this from the Australian Tax Office?", "type": "TAX_INFO"},
            {"question": "What kind of tax form am I looking at?", "type": "TAX_INFO"},
            {"question": "What's the tax period for this document?", "type": "TAX_INFO"},
            {"question": "Can you locate the Australian Business Number?", "type": "TAX_INFO"},
            {"question": "What tax information is contained in this document?", "type": "TAX_INFO"},
            {"question": "Is this a notice of assessment?", "type": "TAX_INFO"},
            {"question": "What tax document has the ATO provided here?", "type": "TAX_INFO"},
            {"question": "When was this tax document issued?", "type": "TAX_INFO"},
            {"question": "What's the document ID on this tax form?", "type": "TAX_INFO"},
            {"question": "Does this document show my tax refund amount?", "type": "TAX_INFO"},
            {"question": "Is this a BAS statement?", "type": "TAX_INFO"},
            {"question": "What's the ATO reference number on this document?", "type": "TAX_INFO"},
        ]
        
        # Add variations with different phrasing for each type to further increase diversity
        phrase_variations = [
            # Document type variations
            {"question": "Please identify what kind of document this is.", "type": "DOCUMENT_TYPE"},
            {"question": "Just tell me if this is a receipt or not.", "type": "DOCUMENT_TYPE"},
            {"question": "I need to know what type of document I'm looking at.", "type": "DOCUMENT_TYPE"},
            {"question": "Could you please identify this document type?", "type": "DOCUMENT_TYPE"},
            {"question": "I'm wondering what kind of document this is.", "type": "DOCUMENT_TYPE"},
            
            # Counting variations
            {"question": "I need to know how many receipts are in this image.", "type": "COUNTING"},
            {"question": "Please tell me the number of receipts shown.", "type": "COUNTING"},
            {"question": "Could you count these receipts for me?", "type": "COUNTING"},
            {"question": "I'd like to know how many receipts are visible.", "type": "COUNTING"},
            {"question": "Just tell me the receipt count please.", "type": "COUNTING"},
            
            # Detail extraction variations
            {"question": "I need to know where this receipt is from.", "type": "DETAIL_EXTRACTION"},
            {"question": "Please tell me what the date is on this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Could you read what items were purchased?", "type": "DETAIL_EXTRACTION"},
            {"question": "I'd like to know what products are listed here.", "type": "DETAIL_EXTRACTION"},
            {"question": "Can you find the store name on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "I'm trying to figure out what was purchased on this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Could you tell me what items this person bought?", "type": "DETAIL_EXTRACTION"},
            {"question": "I'd like to know the name of the store on this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Would you mind telling me the date on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "Can you extract the total amount from this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "Please identify what items were purchased.", "type": "DETAIL_EXTRACTION"},
            {"question": "I'm interested in the purchase date on this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Do me a favor and tell me what store this receipt is from.", "type": "DETAIL_EXTRACTION"},
            {"question": "I'd really appreciate if you could tell me the transaction amount.", "type": "DETAIL_EXTRACTION"},
            {"question": "I need you to extract the purchase details from this receipt.", "type": "DETAIL_EXTRACTION"},
            
            # Payment info variations
            {"question": "I need to confirm how this was paid for.", "type": "PAYMENT_INFO"},
            {"question": "Please check what payment method was used.", "type": "PAYMENT_INFO"},
            {"question": "Could you tell me if this was paid with cash?", "type": "PAYMENT_INFO"},
            {"question": "I'd like to know which payment type was used.", "type": "PAYMENT_INFO"},
            {"question": "Can you find how the payment was processed?", "type": "PAYMENT_INFO"},
            
            # Tax info variations
            {"question": "I need to check what tax form this is.", "type": "TAX_INFO"},
            {"question": "Please find the ABN on this tax document.", "type": "TAX_INFO"},
            {"question": "Could you tell me which tax year this covers?", "type": "TAX_INFO"},
            {"question": "I'd like to know if this is an official ATO document.", "type": "TAX_INFO"},
            {"question": "Can you find the tax file number on this?", "type": "TAX_INFO"},
        ]
        
        # Add the variations to the main list
        questions.extend(phrase_variations)
        
        # Add complex sentence structures and questions with adverbs
        complex_questions = [
            {"question": "Quickly identify whether this is a receipt or tax document.", "type": "DOCUMENT_TYPE"},
            {"question": "Carefully count how many separate receipts appear in this image.", "type": "COUNTING"},
            {"question": "Thoroughly examine this receipt and tell me what store it's from.", "type": "DETAIL_EXTRACTION"},
            {"question": "Briefly explain how this purchase was paid for based on the receipt.", "type": "PAYMENT_INFO"},
            {"question": "Precisely identify which tax year this ATO document pertains to.", "type": "TAX_INFO"},
        ]
        
        # Add additional complex DETAIL_EXTRACTION questions
        complex_detail_questions = [
            {"question": "Meticulously extract all purchase information from this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Carefully identify the store name and total amount on this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Thoroughly analyze this receipt and tell me when it was issued.", "type": "DETAIL_EXTRACTION"},
            {"question": "Precisely determine what items were purchased based on the receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Rapidly identify the transaction date from this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "In detail, list all the items shown on this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Completely examine this receipt and tell me the store's name.", "type": "DETAIL_EXTRACTION"},
            {"question": "Given this receipt, identify exactly what time the purchase occurred.", "type": "DETAIL_EXTRACTION"},
            {"question": "Analyze this receipt and extract the subtotal amount.", "type": "DETAIL_EXTRACTION"},
            {"question": "From this receipt, determine precisely how many items were purchased.", "type": "DETAIL_EXTRACTION"},
        ]
        
        # Add questions with challenging syntax and Australian context for DETAIL_EXTRACTION
        australian_detail_questions = [
            {"question": "What's the GST component on this receipt, mate?", "type": "DETAIL_EXTRACTION"},
            {"question": "Which items in the docket have the highest price?", "type": "DETAIL_EXTRACTION"},
            {"question": "Does this receipt show any specials or markdowns?", "type": "DETAIL_EXTRACTION"},
            {"question": "Can you tell me if there's a loyalty number on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the ABN listed on this till receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "Does the receipt show what aisle these items were from?", "type": "DETAIL_EXTRACTION"},
            {"question": "Is there any mention of reward points on this receipt?", "type": "DETAIL_EXTRACTION"},
            {"question": "What's the total including the GST on this docket?", "type": "DETAIL_EXTRACTION"},
            {"question": "Tell me what department store issued this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "What was the price of groceries on this receipt?", "type": "DETAIL_EXTRACTION"},
        ]
        
        # Add all detail questions to the collection
        complex_questions.extend(complex_detail_questions)
        complex_questions.extend(australian_detail_questions)
        
        questions.extend(complex_questions)
        
        # Shuffle the questions to avoid training bias
        import random
        random.shuffle(questions)
        
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