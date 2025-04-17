"""
Question classifier for multimodal vision-language tasks.

This module implements a classifier to understand different types of questions
about receipts and tax documents.
"""
import os
import torch
import torch.nn as nn
import logging
from transformers import AutoModel, AutoTokenizer


class QuestionClassifier(nn.Module):
    """
    Question classifier for distinguishing different types of questions.
    
    Uses a pre-trained language model to classify questions into predefined
    categories such as document type, counting, detail extraction, etc.
    """
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",  # Changed default model to distilbert
        hidden_size: int = 768, 
        num_classes: int = 5,
        device: str = None,
        use_custom_path: bool = False  # Disabled custom path by default
    ):
        """
        Initialize question classifier.
        
        Args:
            model_name: Pretrained model name or path for the encoder
            hidden_size: Size of encoder hidden states
            num_classes: Number of question categories
            device: Device to run the model on
            use_custom_path: Whether to check for and use the custom path
        """
        super().__init__()
        logger = logging.getLogger(__name__)
        
        # The model_name parameter should contain the path from the config file when use_custom_path is True
        # No hard-coded paths - use only what's provided in the config
        
        # Determine the model path to use
        if use_custom_path:
            # When use_custom_path is True, model_name should be a valid path from config
            if os.path.exists(model_name):
                logger.info(f"Using model from config-provided path: {model_name}")
                model_path = model_name
            else:
                # Path doesn't exist - this is a fatal error since we're expecting a valid path
                error_msg = f"Model path from config does not exist: {model_name}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        else:
            # Using regular HuggingFace model (should not happen in production)
            logger.warning(f"Using HuggingFace model by name (not recommended): {model_name}")
            model_path = model_name
        
        # Load tokenizer FIRST - this ensures consistent vocabularies
        if use_custom_path:
            # Load from local path with appropriate settings
            try:
                logger.info(f"Loading tokenizer from custom path: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                logger.info("Successfully loaded tokenizer from custom path")
            except Exception as e:
                # Don't silently fail - raise the error
                logger.error(f"Failed to load tokenizer from custom path '{model_path}': {e}")
                raise RuntimeError(f"Failed to load tokenizer from custom path '{model_path}': {e}")
        else:
            # Regular loading from HuggingFace
            try:
                logger.info(f"Loading tokenizer from HuggingFace: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"Successfully loaded tokenizer: {model_name}")
            except Exception as e:
                # Don't silently fail - raise the error
                logger.error(f"Failed to load tokenizer '{model_name}' from HuggingFace: {e}")
                raise RuntimeError(f"Failed to load tokenizer '{model_name}' from HuggingFace: {e}")
        
        # Initialize encoder with appropriate settings
        if use_custom_path:
            # Load from local path with appropriate settings
            try:
                logger.info(f"Loading model from custom path: {model_path}")
                # Check if directory exists
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Custom model path does not exist: {model_path}")
                
                # Check if config.json exists
                if not os.path.exists(os.path.join(model_path, "config.json")):
                    raise FileNotFoundError(f"No config.json found in custom model path: {model_path}")
                
                # Load model
                self.encoder = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded model from custom path. Type: {type(self.encoder).__name__}")
            except Exception as e:
                logger.error(f"Failed to load model from custom path '{model_path}': {e}")
                raise RuntimeError(f"Failed to load model from custom path '{model_path}': {e}")
        else:
            # Regular loading from HuggingFace
            try:
                logger.info(f"Loading model from HuggingFace: {model_name}")
                self.encoder = AutoModel.from_pretrained(model_name)
                logger.info(f"Successfully loaded model from HuggingFace. Type: {type(self.encoder).__name__}")
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}' from HuggingFace: {e}")
                raise RuntimeError(f"Failed to load model '{model_name}' from HuggingFace: {e}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size//2, num_classes)
        )
        
        # Class mapping
        self.question_classes = {
            0: "DOCUMENT_TYPE",
            1: "COUNTING",
            2: "DETAIL_EXTRACTION",
            3: "PAYMENT_INFO",
            4: "TAX_INFO"
        }
        
        # Set device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the question classifier.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Logits for question classification
        """
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(pooled_output)
        return logits
    
    def predict_question_type(self, question):
        """
        Classify a question string into its type.
        
        Args:
            question: Question text string
            
        Returns:
            Predicted question type as string
            
        Raises:
            RuntimeError: If the model or tokenizer fails to process the question
        """
        logger = logging.getLogger(__name__)
        
        # Get vocabulary size from tokenizer
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is None:
            # Try to determine vocab size from model config
            if hasattr(self.encoder, "config") and hasattr(self.encoder.config, "vocab_size"):
                vocab_size = self.encoder.config.vocab_size
            else:
                # Raise error - we need to know the vocab size
                raise RuntimeError("Could not determine vocabulary size from tokenizer or model config")
                
        logger.debug(f"Tokenizer vocabulary size: {vocab_size}")
        
        # Process input with tokenizer
        inputs = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Check token IDs against vocabulary size
        max_token_id = inputs.input_ids.max().item()
        if max_token_id >= vocab_size:
            error_msg = f"Token ID out of vocabulary range: max_id={max_token_id}, vocab_size={vocab_size}"
            logger.error(error_msg)
            raise IndexError(error_msg)
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self(inputs.input_ids, inputs.attention_mask)
            pred = torch.argmax(logits, dim=1).item()
        
        # Get class from prediction
        if pred not in self.question_classes:
            error_msg = f"Invalid prediction index: {pred}, valid classes: {list(self.question_classes.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        return self.question_classes[pred]