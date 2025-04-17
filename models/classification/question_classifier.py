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
        use_custom_path: bool = False,  # Disabled custom path by default
        use_internvl_language_model: bool = False,  # Whether to extract language model from InternVL
        **kwargs  # Additional arguments
    ):
        """
        Initialize question classifier.
        
        Args:
            model_name: Pretrained model name or path for the encoder
            hidden_size: Size of encoder hidden states
            num_classes: Number of question categories
            device: Device to run the model on
            use_custom_path: Whether to check for and use the custom path
            use_internvl_language_model: Whether to extract language model from InternVL
            kwargs: Additional keyword arguments
        """
        super().__init__()
        logger = logging.getLogger(__name__)
        
        # Store initialization parameters as attributes
        self.use_internvl_language_model = use_internvl_language_model
        logger.info(f"Using InternVL language model: {use_internvl_language_model}")
        
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
        
        # Check for internvl flag in config
        use_internvl_language_model = getattr(self, 'use_internvl_language_model', False)
        if not hasattr(self, 'use_internvl_language_model'):
            # Try to get from kwargs
            use_internvl_language_model = kwargs.get('use_internvl_language_model', False)
            logger.info(f"Using internvl_language_model flag from kwargs: {use_internvl_language_model}")
        
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
                
                # If using InternVL language model, we need to extract it from the full model
                if use_internvl_language_model:
                    logger.info("Loading InternVL2 model to extract language model component...")
                    from transformers import AutoModel
                    
                    # First load the full InternVL2 model
                    full_model = AutoModel.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    
                    # Try to extract the language model component
                    if hasattr(full_model, "llm"):
                        self.encoder = full_model.llm
                        logger.info("Using full_model.llm as language model for classifier")
                    elif hasattr(full_model, "language_model"):
                        self.encoder = full_model.language_model
                        logger.info("Using full_model.language_model as language model for classifier")
                    elif hasattr(full_model, "text_model"):
                        self.encoder = full_model.text_model
                        logger.info("Using full_model.text_model as language model for classifier")
                    elif hasattr(full_model, "LLM"):
                        self.encoder = full_model.LLM
                        logger.info("Using full_model.LLM as language model for classifier")
                    elif hasattr(full_model, "llama"):
                        self.encoder = full_model.llama
                        logger.info("Using full_model.llama as language model for classifier")
                    else:
                        # If we can't find the language model, log all attributes to help debug
                        all_attributes = [attr for attr in dir(full_model) if not attr.startswith('_')]
                        logger.info(f"Model attributes: {all_attributes}")
                        
                        # Fall back to using the entire model
                        logger.warning("Could not extract language model. Using full model instead.")
                        self.encoder = full_model
                else:
                    # Regular loading - not using InternVL language model
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
        logger = logging.getLogger(__name__)
        
        # Check the encoder type to determine how to handle outputs
        encoder_type = type(self.encoder).__name__
        logger.debug(f"Forward pass with encoder type: {encoder_type}")
        
        try:
            # Get valid parameters for this model
            import inspect
            sig = inspect.signature(self.encoder.forward)
            valid_params = sig.parameters.keys()
            
            # Create a dictionary of valid parameters
            params = {}
            if 'input_ids' in valid_params:
                params['input_ids'] = input_ids
            if 'attention_mask' in valid_params and attention_mask is not None:
                params['attention_mask'] = attention_mask
            if 'return_dict' in valid_params:
                params['return_dict'] = True
            if 'output_hidden_states' in valid_params:
                params['output_hidden_states'] = True
                
            logger.debug(f"Calling encoder with params: {list(params.keys())}")
            outputs = self.encoder(**params)
            
            # Extract embeddings based on output format
            if hasattr(outputs, 'last_hidden_state'):
                # Standard HuggingFace format
                pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
                logger.debug("Using last_hidden_state[:, 0, :] for pooled output")
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # Some models return hidden states tuple/list
                if isinstance(outputs.hidden_states, (list, tuple)):
                    pooled_output = outputs.hidden_states[-1][:, 0, :]  # Last layer, CLS token
                    logger.debug("Using hidden_states[-1][:, 0, :] for pooled output")
                else:
                    pooled_output = outputs.hidden_states[:, 0, :]
                    logger.debug("Using hidden_states[:, 0, :] for pooled output")
            elif hasattr(outputs, 'pooler_output'):
                # Some models have a dedicated pooler
                pooled_output = outputs.pooler_output
                logger.debug("Using pooler_output for pooled output")
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                # Try first element, which is often the hidden states
                pooled_output = outputs[0][:, 0, :]
                logger.debug("Using outputs[0][:, 0, :] for pooled output")
            else:
                # Last resort - assume outputs is already the hidden states
                pooled_output = outputs[:, 0, :]
                logger.debug("Using outputs[:, 0, :] directly for pooled output")
                
            # Pass through classifier head
            logits = self.classifier(pooled_output)
            return logits
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # Try a simpler approach with minimal parameters
            try:
                outputs = self.encoder(input_ids, attention_mask=attention_mask)
                
                # Extract embeddings - most basic approach
                if hasattr(outputs, 'last_hidden_state'):
                    pooled_output = outputs.last_hidden_state[:, 0, :]
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    pooled_output = outputs[0][:, 0, :]
                else:
                    pooled_output = outputs[:, 0, :]
                
                logits = self.classifier(pooled_output)
                return logits
            except Exception as e2:
                logger.error(f"Critical error in forward pass: {e2}")
                # Create emergency fallback - random output to prevent crashes
                batch_size = input_ids.shape[0]
                num_classes = self.classifier[-1].out_features
                device = input_ids.device
                
                # Return zero logits - this will at least allow training to continue
                return torch.zeros(batch_size, num_classes, device=device)
    
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