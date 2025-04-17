"""
Balanced dataset for training question classifiers with equal representation for all classes.

This module provides a balanced dataset with equal representation of all question types
to address the class imbalance issue identified in training.
"""
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class BalancedQuestionDataset(Dataset):
    """
    Balanced dataset for training question classification models.
    
    Contains carefully balanced labeled questions with equal representation
    for all question types to avoid bias toward any specific class.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        tokenizer_name: str = None,
        max_length: int = 128,
        split: str = "train",
        create_default: bool = True,
    ):
        """
        Initialize balanced question dataset.
        
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
        self.file_path = self.data_dir / f"balanced_question_dataset_{split}.json"
        
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
        logger = logging.getLogger(__name__)
        
        # Verify tokenizer name is provided
        if tokenizer_name is None:
            raise ValueError("tokenizer_name must be provided - no default fallback")
            
        # Check if it's a path to a local tokenizer
        tokenizer_path = Path(tokenizer_name)
        if tokenizer_path.exists() and tokenizer_path.is_dir():
            logger.info(f"Loading tokenizer from local path: {tokenizer_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                logger.info(f"Successfully loaded tokenizer from local path")
            except Exception as e:
                logger.error(f"Failed to load tokenizer from path '{tokenizer_path}': {e}")
                raise RuntimeError(f"Failed to load tokenizer from path '{tokenizer_path}': {e}")
        else:
            # This is a serious error - we should not be loading from HuggingFace in production
            logger.error(f"Tokenizer path does not exist: {tokenizer_name}")
            raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_name}")
            
        # If we reach this point, tokenizer loading failed completely
        if not hasattr(self, 'tokenizer'):
            logger.error("No tokenizer could be loaded")
            raise RuntimeError("Failed to load any tokenizer")
        
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
        Create balanced question dataset with equal representation for all classes.
        
        Returns:
            List of question dictionaries
        """
        # ========== DOCUMENT_TYPE QUESTIONS (80+ examples) ==========
        document_type_questions = [
            # Basic document identification questions
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
            
            # More specific document type questions
            {"question": "Is this an invoice, receipt, or tax form?", "type": "DOCUMENT_TYPE"},
            {"question": "Do you recognize what type of document this is?", "type": "DOCUMENT_TYPE"},
            {"question": "Classify this document for me please.", "type": "DOCUMENT_TYPE"},
            {"question": "Tell me if this is a tax-related document.", "type": "DOCUMENT_TYPE"},
            {"question": "What kind of financial document is shown here?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this an ATO tax document or a shop receipt?", "type": "DOCUMENT_TYPE"},
            {"question": "What's the nature of this document?", "type": "DOCUMENT_TYPE"},
            {"question": "Document type identification needed.", "type": "DOCUMENT_TYPE"},
            {"question": "What sort of record is this?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this a tax return, BAS, or receipt?", "type": "DOCUMENT_TYPE"},
            
            # Questions with qualifiers
            {"question": "Please identify what kind of document this is.", "type": "DOCUMENT_TYPE"},
            {"question": "Just tell me if this is a receipt or not.", "type": "DOCUMENT_TYPE"},
            {"question": "I need to know what type of document I'm looking at.", "type": "DOCUMENT_TYPE"},
            {"question": "Could you please identify this document type?", "type": "DOCUMENT_TYPE"},
            {"question": "I'm wondering what kind of document this is.", "type": "DOCUMENT_TYPE"},
            {"question": "Would it be possible to tell me what kind of document this is?", "type": "DOCUMENT_TYPE"},
            {"question": "I'd appreciate if you could identify this document type.", "type": "DOCUMENT_TYPE"},
            {"question": "Do you mind telling me what type of document is shown?", "type": "DOCUMENT_TYPE"},
            {"question": "If it's not too much trouble, tell me what this document is.", "type": "DOCUMENT_TYPE"},
            {"question": "Quick question - what kind of document am I looking at?", "type": "DOCUMENT_TYPE"},
            
            # Questions with adverbs 
            {"question": "Quickly identify whether this is a receipt or tax document.", "type": "DOCUMENT_TYPE"},
            {"question": "Immediately tell me what kind of document this is.", "type": "DOCUMENT_TYPE"},
            {"question": "Precisely identify this document type.", "type": "DOCUMENT_TYPE"},
            {"question": "Clearly state what kind of document I'm looking at.", "type": "DOCUMENT_TYPE"},
            {"question": "Promptly determine what type of document this is.", "type": "DOCUMENT_TYPE"},
            {"question": "Accurately classify this document.", "type": "DOCUMENT_TYPE"},
            {"question": "Briefly state what this document is.", "type": "DOCUMENT_TYPE"},
            {"question": "Definitely tell me if this is a receipt.", "type": "DOCUMENT_TYPE"},
            {"question": "Concisely identify this document type.", "type": "DOCUMENT_TYPE"},
            {"question": "Swiftly determine what document we're looking at.", "type": "DOCUMENT_TYPE"},
            
            # Australian context questions
            {"question": "Is this a docket from a shop?", "type": "DOCUMENT_TYPE"},
            {"question": "Does this appear to be an ATO notice?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this a GST receipt?", "type": "DOCUMENT_TYPE"},
            {"question": "Can you tell if this is a tax invoice?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this document from Centrelink?", "type": "DOCUMENT_TYPE"},
            {"question": "Does this look like a Medicare document?", "type": "DOCUMENT_TYPE"},
            {"question": "Would you say this is a Council rates notice?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this a receipt from Woolies or Coles?", "type": "DOCUMENT_TYPE"},
            {"question": "Can you tell me if this is a receipt from a service station?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this an EFTPOS receipt?", "type": "DOCUMENT_TYPE"},
            
            # Adversarial questions that appear similar to other categories
            {"question": "What document shows information about my payment?", "type": "DOCUMENT_TYPE"},
            {"question": "What document lists the detailed items?", "type": "DOCUMENT_TYPE"},
            {"question": "Can you identify if this document contains tax information?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this document showing receipt details?", "type": "DOCUMENT_TYPE"},
            {"question": "Does this document look like it's for counting purposes?", "type": "DOCUMENT_TYPE"},
            {"question": "What kind of document shows payment details?", "type": "DOCUMENT_TYPE"},
            {"question": "Identify this financial document.", "type": "DOCUMENT_TYPE"},
            {"question": "What type of document provides transaction details?", "type": "DOCUMENT_TYPE"},
            {"question": "Is this the kind of document that lists tax amounts?", "type": "DOCUMENT_TYPE"},
            {"question": "What document would have store information?", "type": "DOCUMENT_TYPE"},
        ]
        
        # ========== COUNTING QUESTIONS (80+ examples) ==========
        counting_questions = [
            # Basic counting questions
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
            
            # Questions with qualifiers
            {"question": "I need to know how many receipts are in this image.", "type": "COUNTING"},
            {"question": "Please tell me the number of receipts shown.", "type": "COUNTING"},
            {"question": "Could you count these receipts for me?", "type": "COUNTING"},
            {"question": "I'd like to know how many receipts are visible.", "type": "COUNTING"},
            {"question": "Just tell me the receipt count please.", "type": "COUNTING"},
            {"question": "Would you mind counting the receipts?", "type": "COUNTING"},
            {"question": "If you could count the receipts, I'd appreciate it.", "type": "COUNTING"},
            {"question": "I'm wondering how many receipts there are.", "type": "COUNTING"},
            {"question": "Do me a favor and count the receipts.", "type": "COUNTING"},
            {"question": "Quick question - how many receipts do you see?", "type": "COUNTING"},
            
            # Questions with adverbs
            {"question": "Carefully count how many separate receipts appear in this image.", "type": "COUNTING"},
            {"question": "Precisely count the number of receipts present.", "type": "COUNTING"},
            {"question": "Quickly tell me how many receipts you can see.", "type": "COUNTING"},
            {"question": "Accurately determine the number of receipts.", "type": "COUNTING"},
            {"question": "Thoroughly check how many receipts there are.", "type": "COUNTING"},
            {"question": "Definitely state the total receipt count.", "type": "COUNTING"},
            {"question": "Clearly state how many receipts you observe.", "type": "COUNTING"},
            {"question": "Attentively count all receipts in the image.", "type": "COUNTING"},
            {"question": "Promptly tell me the receipt count.", "type": "COUNTING"},
            {"question": "Meticulously count each receipt shown.", "type": "COUNTING"},
            
            # Questions about tax documents
            {"question": "How many tax documents are in this image?", "type": "COUNTING"},
            {"question": "Count the number of tax forms shown.", "type": "COUNTING"},
            {"question": "Are there multiple tax documents here?", "type": "COUNTING"},
            {"question": "How many separate tax forms can you identify?", "type": "COUNTING"},
            {"question": "Tell me how many ATO documents are shown.", "type": "COUNTING"},
            {"question": "Is there more than one tax document?", "type": "COUNTING"},
            {"question": "How many distinct tax forms do you see?", "type": "COUNTING"},
            {"question": "What's the total number of tax documents?", "type": "COUNTING"},
            {"question": "Count how many tax forms are present.", "type": "COUNTING"},
            {"question": "Are there multiple financial documents visible?", "type": "COUNTING"},
            
            # Australian context questions
            {"question": "How many dockets are in this image?", "type": "COUNTING"},
            {"question": "Count the number of shop dockets shown.", "type": "COUNTING"},
            {"question": "How many separate EFTPOS receipts can you see?", "type": "COUNTING"},
            {"question": "Count how many Centrelink letters are visible.", "type": "COUNTING"},
            {"question": "How many ATO notices do you see?", "type": "COUNTING"},
            {"question": "Are there multiple GST receipts here?", "type": "COUNTING"},
            {"question": "Tell me how many Woolies receipts are shown.", "type": "COUNTING"},
            {"question": "How many tax invoices can you count?", "type": "COUNTING"},
            {"question": "Count the number of Medicare documents visible.", "type": "COUNTING"},
            {"question": "How many separate servo receipts are in this image?", "type": "COUNTING"},
            
            # Adversarial questions that appear similar to other categories
            {"question": "Count the document types in this image.", "type": "COUNTING"},
            {"question": "How many different payment methods are shown?", "type": "COUNTING"},
            {"question": "Count how many details are visible on the receipt.", "type": "COUNTING"},
            {"question": "How many tax-related documents are there?", "type": "COUNTING"},
            {"question": "Count the number of documents with payment info.", "type": "COUNTING"},
            {"question": "How many financial documents are present?", "type": "COUNTING"},
            {"question": "Count the receipts that show payment methods.", "type": "COUNTING"},
            {"question": "How many receipts with detailed information can you count?", "type": "COUNTING"},
            {"question": "Count the number of documents with tax information.", "type": "COUNTING"},
            {"question": "How many receipts containing store details are visible?", "type": "COUNTING"},
        ]
        
        # ========== DETAIL_EXTRACTION QUESTIONS (80+ examples) ==========
        detail_extraction_questions = [
            # Basic detail extraction questions
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
            
            # Specific receipt detail questions
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
            
            # Questions with qualifiers
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
            
            # Questions with adverbs
            {"question": "Thoroughly examine this receipt and tell me what store it's from.", "type": "DETAIL_EXTRACTION"},
            {"question": "Meticulously extract all purchase information from this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Carefully identify the store name and total amount on this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Thoroughly analyze this receipt and tell me when it was issued.", "type": "DETAIL_EXTRACTION"},
            {"question": "Precisely determine what items were purchased based on the receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Rapidly identify the transaction date from this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "In detail, list all the items shown on this receipt.", "type": "DETAIL_EXTRACTION"},
            {"question": "Completely examine this receipt and tell me the store's name.", "type": "DETAIL_EXTRACTION"},
            {"question": "Given this receipt, identify exactly what time the purchase occurred.", "type": "DETAIL_EXTRACTION"},
            {"question": "Analyze this receipt and extract the subtotal amount.", "type": "DETAIL_EXTRACTION"},
            
            # Australian context questions
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
        
        # ========== PAYMENT_INFO QUESTIONS (80+ examples) ==========
        payment_info_questions = [
            # Basic payment questions
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
            
            # Questions with qualifiers
            {"question": "I need to confirm how this was paid for.", "type": "PAYMENT_INFO"},
            {"question": "Please check what payment method was used.", "type": "PAYMENT_INFO"},
            {"question": "Could you tell me if this was paid with cash?", "type": "PAYMENT_INFO"},
            {"question": "I'd like to know which payment type was used.", "type": "PAYMENT_INFO"},
            {"question": "Can you find how the payment was processed?", "type": "PAYMENT_INFO"},
            {"question": "Would you mind checking how this was paid for?", "type": "PAYMENT_INFO"},
            {"question": "I'm curious about the payment method used here.", "type": "PAYMENT_INFO"},
            {"question": "If possible, tell me how this person paid.", "type": "PAYMENT_INFO"},
            {"question": "Do you see what payment type was used?", "type": "PAYMENT_INFO"},
            {"question": "Can you figure out how they paid?", "type": "PAYMENT_INFO"},
            
            # Questions with adverbs
            {"question": "Briefly explain how this purchase was paid for based on the receipt.", "type": "PAYMENT_INFO"},
            {"question": "Quickly identify the payment method shown on this receipt.", "type": "PAYMENT_INFO"},
            {"question": "Clearly indicate how this transaction was paid for.", "type": "PAYMENT_INFO"},
            {"question": "Specifically tell me what payment type was used.", "type": "PAYMENT_INFO"},
            {"question": "Thoroughly check what method of payment was used.", "type": "PAYMENT_INFO"},
            {"question": "Precisely identify the payment method used here.", "type": "PAYMENT_INFO"},
            {"question": "Carefully determine how this purchase was paid for.", "type": "PAYMENT_INFO"},
            {"question": "Accurately tell me the payment method for this transaction.", "type": "PAYMENT_INFO"},
            {"question": "Promptly identify the payment type from this receipt.", "type": "PAYMENT_INFO"},
            {"question": "Definitively state how this purchase was paid for.", "type": "PAYMENT_INFO"},
            
            # Questions about payment details
            {"question": "Was the payment processed successfully?", "type": "PAYMENT_INFO"},
            {"question": "What were the last four digits of the card used?", "type": "PAYMENT_INFO"},
            {"question": "Did the payment go through in one transaction?", "type": "PAYMENT_INFO"},
            {"question": "Was the payment declined initially?", "type": "PAYMENT_INFO"},
            {"question": "Was the payment split between multiple methods?", "type": "PAYMENT_INFO"},
            {"question": "Does it mention the payment terminal used?", "type": "PAYMENT_INFO"},
            {"question": "What was the payment processing fee, if any?", "type": "PAYMENT_INFO"},
            {"question": "Did they pay using contactless or chip and PIN?", "type": "PAYMENT_INFO"},
            {"question": "Is there an authorization code for the payment?", "type": "PAYMENT_INFO"},
            {"question": "Does it indicate if the payment was approved?", "type": "PAYMENT_INFO"},
            
            # Australian context questions
            {"question": "Did they tap or insert their card to pay?", "type": "PAYMENT_INFO"},
            {"question": "Was this paid for with PayID?", "type": "PAYMENT_INFO"},
            {"question": "Did they use PayPass to make this payment?", "type": "PAYMENT_INFO"},
            {"question": "Was this a BPAY payment?", "type": "PAYMENT_INFO"},
            {"question": "Did they use Afterpay for this purchase?", "type": "PAYMENT_INFO"},
            {"question": "Was a flybuys card used with this payment?", "type": "PAYMENT_INFO"},
            {"question": "Did they pay using Zip Pay?", "type": "PAYMENT_INFO"},
            {"question": "Was this purchase made with a Commonwealth Bank card?", "type": "PAYMENT_INFO"},
            {"question": "Does it show if they paid using Osko?", "type": "PAYMENT_INFO"},
            {"question": "Did they use Westpac PayWear to pay?", "type": "PAYMENT_INFO"},
            
            # Adversarial questions that appear similar to other categories
            {"question": "What details are shown about the payment?", "type": "PAYMENT_INFO"},
            {"question": "Extract the payment information from this receipt.", "type": "PAYMENT_INFO"},
            {"question": "What does this document say about how it was paid for?", "type": "PAYMENT_INFO"},
            {"question": "Count the payment methods used for this purchase.", "type": "PAYMENT_INFO"},
            {"question": "What's the payment-related tax information?", "type": "PAYMENT_INFO"},
            {"question": "Find out how much of the payment was tax.", "type": "PAYMENT_INFO"},
            {"question": "Can you identify the payment section on this document?", "type": "PAYMENT_INFO"},
            {"question": "Tell me about the payment details, not the items.", "type": "PAYMENT_INFO"},
            {"question": "What type of payment was processed for this document?", "type": "PAYMENT_INFO"},
            {"question": "Is the payment method shown on this type of document?", "type": "PAYMENT_INFO"},
        ]
        
        # ========== TAX_INFO QUESTIONS (80+ examples) ==========
        tax_info_questions = [
            # Basic tax document questions
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
            
            # Questions with qualifiers
            {"question": "I need to check what tax form this is.", "type": "TAX_INFO"},
            {"question": "Please find the ABN on this tax document.", "type": "TAX_INFO"},
            {"question": "Could you tell me which tax year this covers?", "type": "TAX_INFO"},
            {"question": "I'd like to know if this is an official ATO document.", "type": "TAX_INFO"},
            {"question": "Can you find the tax file number on this?", "type": "TAX_INFO"},
            {"question": "Would you mind checking what tax form this is?", "type": "TAX_INFO"},
            {"question": "I'm wondering if this is a tax assessment notice.", "type": "TAX_INFO"},
            {"question": "If possible, tell me what tax document this is.", "type": "TAX_INFO"},
            {"question": "I'd appreciate if you could identify this tax form.", "type": "TAX_INFO"},
            {"question": "Do you see what ATO document this is?", "type": "TAX_INFO"},
            
            # Questions with adverbs
            {"question": "Precisely identify which tax year this ATO document pertains to.", "type": "TAX_INFO"},
            {"question": "Carefully determine what type of tax document this is.", "type": "TAX_INFO"},
            {"question": "Quickly identify the ABN on this tax form.", "type": "TAX_INFO"},
            {"question": "Thoroughly examine this ATO document and tell me what it is.", "type": "TAX_INFO"},
            {"question": "Accurately state which financial year this tax document covers.", "type": "TAX_INFO"},
            {"question": "Clearly indicate what tax information this document contains.", "type": "TAX_INFO"},
            {"question": "Specifically tell me what ATO form this is.", "type": "TAX_INFO"},
            {"question": "Meticulously identify the tax file number on this document.", "type": "TAX_INFO"},
            {"question": "Promptly tell me what tax document I'm looking at.", "type": "TAX_INFO"},
            {"question": "Definitively state if this is a BAS statement.", "type": "TAX_INFO"},
            
            # Detailed tax questions
            {"question": "What's my taxable income according to this document?", "type": "TAX_INFO"},
            {"question": "How much GST is reported on this form?", "type": "TAX_INFO"},
            {"question": "What's the total tax withheld amount?", "type": "TAX_INFO"},
            {"question": "Does this show my PAYG installment amount?", "type": "TAX_INFO"},
            {"question": "What's my tax refund amount according to this notice?", "type": "TAX_INFO"},
            {"question": "How much super guarantee is shown on this statement?", "type": "TAX_INFO"},
            {"question": "What tax offset amounts are listed?", "type": "TAX_INFO"},
            {"question": "Does this statement show tax deductions?", "type": "TAX_INFO"},
            {"question": "What's the Medicare levy amount on this assessment?", "type": "TAX_INFO"},
            {"question": "How much income tax was assessed?", "type": "TAX_INFO"},
            
            # Australian tax-specific questions
            {"question": "Is this a HECS/HELP statement?", "type": "TAX_INFO"},
            {"question": "Does this document mention franking credits?", "type": "TAX_INFO"},
            {"question": "Is this a superannuation statement?", "type": "TAX_INFO"},
            {"question": "What's the FBT amount shown on this document?", "type": "TAX_INFO"},
            {"question": "Is this a CGT statement?", "type": "TAX_INFO"},
            {"question": "Does this show PAYG summary information?", "type": "TAX_INFO"},
            {"question": "Is this an activity statement from the ATO?", "type": "TAX_INFO"},
            {"question": "What's the GST registration status shown?", "type": "TAX_INFO"},
            {"question": "Does this document show my superannuation guarantee?", "type": "TAX_INFO"},
            {"question": "Is this an income statement from myGov?", "type": "TAX_INFO"},
            
            # Adversarial questions that appear similar to other categories
            {"question": "What does this tax document say about payment methods?", "type": "TAX_INFO"},
            {"question": "Extract the tax information from this document.", "type": "TAX_INFO"},
            {"question": "Tell me about the tax details, not the payment.", "type": "TAX_INFO"},
            {"question": "Count how many tax items are listed on this form.", "type": "TAX_INFO"},
            {"question": "What type of tax document am I looking at?", "type": "TAX_INFO"},
            {"question": "Find the tax-related payment information.", "type": "TAX_INFO"},
            {"question": "Is this tax document or a payment receipt?", "type": "TAX_INFO"},
            {"question": "What tax details can you extract from this?", "type": "TAX_INFO"},
            {"question": "How many tax-related items are on this document?", "type": "TAX_INFO"},
            {"question": "What tax office issued this document?", "type": "TAX_INFO"},
        ]
        
        # Combine all question types
        all_questions = []
        all_questions.extend(document_type_questions)
        all_questions.extend(counting_questions)
        all_questions.extend(detail_extraction_questions)
        all_questions.extend(payment_info_questions)
        all_questions.extend(tax_info_questions)
        
        # Shuffle the questions to avoid training bias
        import random
        random.shuffle(all_questions)
        
        return all_questions
    
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


def create_balanced_question_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    tokenizer_name: str = None,  # No default - must be provided from config
    max_length: int = 128,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for question classification training with balanced classes.
    
    Args:
        data_dir: Directory containing question data
        batch_size: Batch size for dataloaders
        tokenizer_name: Name of tokenizer to use
        max_length: Maximum sequence length
        num_workers: Number of workers for dataloaders
        
    Returns:
        Dictionary of dataloaders for train, validation, and test splits
    """
    # Create datasets
    train_dataset = BalancedQuestionDataset(
        data_dir=data_dir,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="train",
        create_default=True
    )
    
    val_dataset = BalancedQuestionDataset(
        data_dir=data_dir,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="val",
        create_default=True
    )
    
    test_dataset = BalancedQuestionDataset(
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