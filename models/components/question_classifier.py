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
        model_name: str = "ModernBert-base",
        hidden_size: int = 768, 
        num_classes: int = 5,
        device: str = None,
        use_custom_path: bool = True
    ):
        """
        Initialize question classifier.
        
        Args:
            model_name: Pretrained model name or path for the encoder
            hidden_size: Size of encoder hidden states
            num_classes: Number of question categories
            device: Device to run the model on
            use_custom_path: Whether to check for and use the custom ModernBert path
        """
        super().__init__()
        logger = logging.getLogger(__name__)
        
        # Check for custom ModernBert path
        custom_path = "/home/jovyan/nfs_share/models/huggingface/hub/ModernBERT-base"
        
        # Determine the model path to use
        if use_custom_path and os.path.exists(custom_path) and model_name != custom_path:
            logger.info(f"Found ModernBert at custom path: {custom_path}")
            model_path = custom_path
        elif model_name.startswith("/") and os.path.exists(model_name):
            # If model_name is already a full path and exists, use it
            logger.info(f"Using model from provided absolute path: {model_name}")
            model_path = model_name
            use_custom_path = True  # Treat absolute paths like custom paths for loading
        else:
            logger.info(f"Using model from provided name: {model_name}")
            model_path = model_name
            use_custom_path = False  # Reset for model name (not path)
        
        # Initialize encoder with appropriate settings
        try:
            if use_custom_path:
                # Load from local path with appropriate settings
                self.encoder = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                logger.info("Successfully loaded model from custom path")
            else:
                # Regular loading
                self.encoder = AutoModel.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Error loading encoder model: {e}")
            logger.info("Falling back to default model")
            self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size//2, num_classes)
        )
        
        # Load tokenizer with appropriate settings
        try:
            if use_custom_path:
                # Load from local path with appropriate settings
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                logger.info("Successfully loaded tokenizer from custom path")
            else:
                # Regular loading
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Error loading tokenizer: {e}")
            logger.info("Falling back to default tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
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
        """
        inputs = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            logits = self(inputs.input_ids, inputs.attention_mask)
            pred = torch.argmax(logits, dim=1).item()
        
        return self.question_classes[pred]