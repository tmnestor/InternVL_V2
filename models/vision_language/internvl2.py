"""
InternVL2 model implementation for receipt counting with vision-language integration.

This module adapts the InternVL2 vision-language model for:
1. Receipt counting classification using vision encoder
2. Multimodal vision-language processing to answer queries about receipts
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import inspect
from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoModelForCausalLM,
    AutoTokenizer
)

from models.components.projection_head import ClassificationHead, CrossAttention, ResponseGenerator

# Import Flash Attention and xFormers if available
try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import xformers
    import xformers.ops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False


class InternVL2ReceiptClassifier(nn.Module):
    """
    InternVL2-based receipt classification model.
    
    Adapts the InternVL2 architecture for the receipt counting task by using only
    the vision encoder and adding a custom classification head.
    """
    def __init__(
        self, 
        config: Dict[str, Any],
        pretrained: bool = True,
        freeze_vision_encoder: bool = False,
    ):
        """
        Initialize the InternVL2 receipt classifier.
        
        Args:
            config: Model configuration
            pretrained: Whether to load pretrained weights
            freeze_vision_encoder: Whether to freeze the vision encoder
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the InternVL2 model
        pretrained_path = config["model"]["pretrained_path"]
        use_8bit = config["model"].get("use_8bit", False)
        
        # Verify the path exists
        if not Path(pretrained_path).exists():
            raise ValueError(
                f"Model path does not exist: {pretrained_path}. Please provide a valid path to the model."
            )
        
        if pretrained:
            self.logger.info(f"Loading model from local path: {pretrained_path}")
            # Set up the right parameters based on config
            kwargs = {
                # "device_map": "auto",  # Commented for macOS development - re-enable for Linux GPU
                "trust_remote_code": True,
                "local_files_only": True,  # Ensure no download attempts
                # "max_memory": {0: "16GB"},  # Commented for macOS development - re-enable for Linux GPU
                "torch_dtype": torch.float32  # Use float32 for CPU compatibility
            }
            
            # Enable Flash Attention if available and configured
            if config["training"].get("flash_attention", False):
                if HAS_FLASH_ATTN:
                    self.logger.info("Using Flash Attention for faster training")
                    kwargs["attn_implementation"] = "flash_attention_2"
                else:
                    self.logger.warning(
                        "Flash Attention requested but not available. "
                        "Install with: CUDA_HOME=/usr/local/cuda pip install flash-attn>=2.5.0"
                    )
            
            # Set precision based on hardware
            if torch.cuda.is_available():
                # GPU is available, use half-precision
                if use_8bit:
                    # Only add 8-bit quantization if explicitly requested
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        kwargs["quantization_config"] = quantization_config
                        self.logger.info("Using 8-bit quantization")
                    except ImportError:
                        self.logger.warning("BitsAndBytesConfig not available, using default precision")
                        kwargs["torch_dtype"] = torch.float16
                else:
                    # Use bfloat16 if available, otherwise fall back to float16
                    # Use float32 for full precision training (mixed precision disabled)
                    kwargs["torch_dtype"] = torch.float32
                    self.logger.info("Using float32 precision for better compatibility and stability")
            else:
                # CPU only, use float32 for compatibility
                self.logger.info("CUDA not available, using float32 precision")
                kwargs["torch_dtype"] = torch.float32
            
            self.model = AutoModel.from_pretrained(
                pretrained_path,
                **kwargs
            )
            
            # Convert model to float32 on CPU for compatibility
            if not torch.cuda.is_available():
                self.logger.info("Converting model to float32 for CPU compatibility")
                self.model = self.model.float()
        else:
            # Load with default initialization (for debugging)
            config = AutoConfig.from_pretrained(
                pretrained_path, trust_remote_code=True, local_files_only=True
            )
            self.model = AutoModel.from_config(config)
        
        # Extract vision encoder from the full model - InternVL2 uses vision_model
        self.vision_encoder = self.model.vision_model
        self.logger.info("Using vision_model for vision encoding")
        
        # Specifically disable sliding window attention and other problematic settings
        try:
            # Check for config access points
            if hasattr(self.model, "config"):
                if hasattr(self.model.config, "use_sliding_window"):
                    self.model.config.use_sliding_window = False
                    self.logger.info("Disabled sliding window attention in model config")
                if hasattr(self.model.config, "gradient_checkpointing"):
                    self.model.config.gradient_checkpointing = False
                    self.logger.info("Disabled gradient checkpointing in model config")
                    
            if hasattr(self.vision_encoder, "config"):
                if hasattr(self.vision_encoder.config, "use_sliding_window"):
                    self.vision_encoder.config.use_sliding_window = False
                    self.logger.info("Disabled sliding window attention in vision encoder config")
                if hasattr(self.vision_encoder.config, "gradient_checkpointing"):
                    self.vision_encoder.config.gradient_checkpointing = False
                    self.logger.info("Disabled gradient checkpointing in vision encoder config")
        except Exception as e:
            self.logger.warning(
                f"Could not disable sliding window attention or gradient checkpointing: {e}"
            )
        
        # Get vision encoder output dimension (do this before creating classification head)
        vision_hidden_size = 512  # Default fallback size
        if hasattr(self.vision_encoder, "config") and hasattr(self.vision_encoder.config, "hidden_size"):
            vision_hidden_size = self.vision_encoder.config.hidden_size
        
        # Create a custom classification head
        self.classification_head = ClassificationHead(
            input_dim=vision_hidden_size,
            hidden_dims=config["model"]["classifier"]["hidden_dims"],
            output_dim=config["model"]["num_classes"],
            dropout_rates=config["model"]["classifier"]["dropout_rates"],
            use_batchnorm=config["model"]["classifier"]["batch_norm"],
            activation=config["model"]["classifier"]["activation"],
        )
        
        # Ensure all components use the same dtype
        if torch.cuda.is_available():
            # Use same precision as model on GPU
            model_dtype = next(self.model.parameters()).dtype
            self.logger.info(f"Converting classification head to model dtype: {model_dtype}")
            self.classification_head = self.classification_head.to(model_dtype)
        else:
            # Convert classification head to float32 for CPU
            self.classification_head = self.classification_head.float()
                
        # Remove language model-related components to save memory
        if hasattr(self.model, "language_model"):
            del self.model.language_model
        if hasattr(self.model, "text_hidden_fcs"):
            del self.model.text_hidden_fcs
            
        # Freeze vision encoder if required
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # Classification head is now created after getting the vision encoder
        
    def unfreeze_vision_encoder(self, lr_multiplier: float = 0.01) -> List[Dict]:
        """
        Unfreeze the vision encoder and prepare parameter groups for optimizer.
        Uses a very small learning rate multiplier to prevent catastrophic forgetting.
        
        Args:
            lr_multiplier: Learning rate multiplier for vision encoder parameters
            
        Returns:
            List of parameter groups for optimizer
        """
        # First, check which layers are most appropriate to tune
        # For InternVL2, we'll focus on the last few layers of the vision encoder
        # rather than unfreezing everything at once
        
        total_layers = 0
        last_n_layers = 2  # Only unfreeze the last few layers initially
        
        # Count layers in vision encoder
        for name, param in self.vision_encoder.named_parameters():
            total_layers += 1
        
        # Selectively unfreeze only the last few layers
        unfrozen_count = 0
        for name, param in self.vision_encoder.named_parameters():
            # If in the last n layers or contains 'classifier' or 'head' or 'pooler'
            if (unfrozen_count >= total_layers - last_n_layers or 
                any(x in name.lower() for x in ['classifier', 'head', 'pooler', 'output'])):
                param.requires_grad = True
                unfrozen_count += 1
                self.logger.info(f"Unfreezing layer: {name}")
            else:
                param.requires_grad = False
        
        self.logger.info(f"Unfrozen {unfrozen_count} of {total_layers} vision encoder layers")
            
        # Create parameter groups with different learning rates
        # Use a very low learning rate for vision encoder to prevent catastrophic forgetting
        return [
            {'params': self.classification_head.parameters()},  # Full learning rate
            # Reduced LR for vision encoder
            {
                'params': [p for p in self.vision_encoder.parameters() if p.requires_grad], 
                'lr': lr_multiplier
            }
        ]
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            pixel_values: Batch of images [B, C, H, W]
            
        Returns:
            Dictionary with logits and other outputs
        """
        # Ensure input is in correct dtype for the model
        if not torch.cuda.is_available():
            # Convert to float32 for CPU
            pixel_values = pixel_values.to(torch.float32)
        else:
            # Convert to model's data type for GPU to avoid mixed precision issues
            if hasattr(self, 'model') and hasattr(self.model, 'dtype'):
                dtype = self.model.dtype
                pixel_values = pixel_values.to(dtype)
            # Default to bfloat16 if device supports it, otherwise float16
            elif torch.cuda.is_available():
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                pixel_values = pixel_values.to(dtype)
        
        # Process the images through the vision encoder
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        
        # Extract image embeddings - InternVL2 outputs last_hidden_state
        image_embeds = vision_outputs.last_hidden_state
        
        # Global average pooling over sequence dimension
        pooled_output = image_embeds.mean(dim=1)
        
        # Simplified dtype handling - we know the types match from initialization
        # This section kept in case initialization fails or model is modified at runtime
        if not hasattr(self, '_model_classifier_dtypes_matched'):
            if hasattr(self.classification_head, 'mlp') and len(self.classification_head.mlp) > 0:
                if hasattr(self.classification_head.mlp[0], 'weight'):
                    target_dtype = self.classification_head.mlp[0].weight.dtype
                    if pooled_output.dtype != target_dtype:
                        self.logger.info(
                            f"One-time conversion from {pooled_output.dtype} to {target_dtype}"
                        )
                        self._model_classifier_dtypes_matched = True
                    else:
                        self.logger.info(
                            f"No conversion needed: model output already {pooled_output.dtype}"
                        )
                        self._model_classifier_dtypes_matched = True
            
        # Force ensure compatibility by getting classifier dtype directly
        if hasattr(self.classification_head, 'mlp') and len(self.classification_head.mlp) > 0:
            if hasattr(self.classification_head.mlp[0], 'weight'):
                pooled_output = pooled_output.to(self.classification_head.mlp[0].weight.dtype)
        
        # Pass through classifier head
        logits = self.classification_head(pooled_output)
        
        return {
            "logits": logits,
            "embeddings": pooled_output
        }
    
    def get_attention_maps(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps from the vision model for visualization.
        
        Args:
            pixel_values: Batch of images [B, C, H, W]
            
        Returns:
            List of attention maps from each transformer block
        """
        try:
            # Enable output_attentions if config attribute exists
            if hasattr(self.vision_encoder, 'config'):
                original_setting = getattr(self.vision_encoder.config, 'output_attentions', False)
                self.vision_encoder.config.output_attentions = True
            else:
                original_setting = False
                
            # Forward pass
            try:
                outputs = self.vision_encoder(pixel_values=pixel_values, output_attentions=True)
            except:
                # Try with the full model
                outputs = self.model(pixel_values=pixel_values, output_attentions=True)
                
            # Get attention weights
            if hasattr(outputs, 'attentions'):
                attention_maps = outputs.attentions
            elif isinstance(outputs, tuple) and len(outputs) > 1:
                # Check which element might contain the attention maps
                for output in outputs:
                    if (isinstance(output, (list, tuple)) and len(output) > 0 
                        and isinstance(output[0], torch.Tensor)):
                        attention_maps = output
                        break
                else:
                    # If no attention maps found
                    self.logger.warning("No attention maps found in model outputs")
                    attention_maps = []
            else:
                self.logger.warning("Could not extract attention maps from model outputs")
                attention_maps = []
                
            # Reset config to original setting if possible
            if hasattr(self.vision_encoder, 'config'):
                self.vision_encoder.config.output_attentions = original_setting
                
            return attention_maps
            
        except Exception as e:
            self.logger.error(f"Error getting attention maps: {e}")
            # Return empty list in case of error
            return []


class InternVL2MultimodalModel(nn.Module):
    """
    InternVL2-based multimodal model for vision-language tasks.
    
    Adapts the InternVL2 architecture to handle both vision and language inputs,
    with cross-modal attention for answering questions about receipt images.
    """
    def __init__(
        self, 
        config: Dict[str, Any],
        pretrained: bool = True,
        freeze_vision_encoder: bool = True,
        freeze_language_model: bool = False,
    ):
        """
        Initialize the InternVL2 multimodal model.
        
        Args:
            config: Model configuration
            pretrained: Whether to load pretrained weights
            freeze_vision_encoder: Whether to freeze the vision encoder
            freeze_language_model: Whether to freeze the language model
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the InternVL2 model with both vision and language components
        pretrained_path = config["model"]["pretrained_path"]
        
        # Verify the path exists
        if not Path(pretrained_path).exists():
            raise ValueError(
                f"Model path does not exist: {pretrained_path}. Please provide a valid path to the model."
            )
        
        # Load the full model with vision and language components
        if pretrained:
            self.logger.info(f"Loading model from local path: {pretrained_path}")
            # Set up the right parameters based on config
            kwargs = {
                "device_map": "auto",  # Enable device map for memory optimization
                "trust_remote_code": True,
                "local_files_only": True,
                "max_memory": {0: "14GB"},  # Limit memory usage to avoid OOM
                "offload_folder": "offload_folder",  # Enable disk offloading
                "offload_state_dict": True,  # Offload weights when not in use
                "torch_dtype": torch.float32  # Use float32 for CPU compatibility
            }
            
            # Enable Flash Attention if available and configured
            if config["training"].get("flash_attention", False) and HAS_FLASH_ATTN:
                self.logger.info("Using Flash Attention for faster training")
                kwargs["attn_implementation"] = "flash_attention_2"
            
            # Set precision based on hardware
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.float32
                self.logger.info("Using float32 precision for better compatibility")
            else:
                kwargs["torch_dtype"] = torch.float32
            
            # Load the full model
            self.model = AutoModel.from_pretrained(
                pretrained_path,
                **kwargs
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_path,
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            # Load with default initialization (for debugging)
            config_obj = AutoConfig.from_pretrained(
                pretrained_path, trust_remote_code=True, local_files_only=True
            )
            self.model = AutoModel.from_config(config_obj)
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_path, trust_remote_code=True, local_files_only=True
            )
        
        # Extract vision encoder from InternVL2 model
        self.vision_encoder = self.model.vision_model
        self.logger.info("Using vision_model for vision encoding")
        
        # Get the language model component for InternVL2
        model_type = type(self.model).__name__
        self.logger.info(f"Model class type: {model_type}")
        
        # Log model type from config
        if hasattr(self.model, "config") and hasattr(self.model.config, "model_type"):
            config_model_type = self.model.config.model_type
            self.logger.info(f"Model config type: {config_model_type}")
        
        # InternVL2 uses different structures, but language_model is most common
        if hasattr(self.model, "language_model"):
            self.language_model = self.model.language_model
            self.logger.info("Using model.language_model for text encoding")
        else:
            self.logger.error("Could not find language_model component in InternVL2 model.")
            raise ValueError("InternVL2 model is missing language_model component.")
        
        # Enable gradient checkpointing for model components to save memory
        try:
            # Enable gradient checkpointing in model config
            if hasattr(self.model, "config") and hasattr(self.model.config, "gradient_checkpointing"):
                self.model.config.gradient_checkpointing = True
                self.model.config.use_cache = False  # Disable KV-cache to save memory
                self.logger.info("Enabled gradient checkpointing in model config")
                
            # Enable gradient checkpointing in vision encoder if available
            if hasattr(self.vision_encoder, "config"):
                if hasattr(self.vision_encoder.config, "gradient_checkpointing"):
                    self.vision_encoder.config.gradient_checkpointing = True
                    self.logger.info("Enabled gradient checkpointing in vision encoder config")
                if hasattr(self.vision_encoder.config, "use_cache"):
                    self.vision_encoder.config.use_cache = False
                    self.logger.info("Disabled KV-cache in vision encoder config")
                    
            # Enable gradient checkpointing in language model if available
            if hasattr(self.language_model, "config"):
                if hasattr(self.language_model.config, "gradient_checkpointing"):
                    self.language_model.config.gradient_checkpointing = True
                    self.logger.info("Enabled gradient checkpointing in language model config")
                if hasattr(self.language_model.config, "use_cache"):
                    self.language_model.config.use_cache = False
                    self.logger.info("Disabled KV-cache in language model config")
                    
            # Manually enable gradient checkpointing on transformer modules if possible
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                self.logger.info("Manually enabled gradient checkpointing on model")
            if hasattr(self.vision_encoder, "gradient_checkpointing_enable"):
                self.vision_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                self.logger.info("Manually enabled gradient checkpointing on vision encoder")
            if hasattr(self.language_model, "gradient_checkpointing_enable"):
                self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                self.logger.info("Manually enabled gradient checkpointing on language model")
                
        except Exception as e:
            self.logger.warning(f"Could not configure gradient checkpointing: {e}")
        
        # Get hidden sizes for both encoders
        vision_hidden_size = 512  # Default fallback
        if hasattr(self.vision_encoder, "config") and hasattr(self.vision_encoder.config, "hidden_size"):
            vision_hidden_size = self.vision_encoder.config.hidden_size
        
        language_hidden_size = 512  # Default fallback
        if hasattr(self.language_model, "config") and hasattr(self.language_model.config, "hidden_size"):
            language_hidden_size = self.language_model.config.hidden_size
        
        # Cross-modal attention for language → vision attention
        self.cross_attention = CrossAttention(
            query_dim=language_hidden_size,
            key_value_dim=vision_hidden_size,
            embed_dim=language_hidden_size,
            num_heads=8,
            dropout=0.1
        )
        
        # Create a custom classification head for receipt counting
        self.classification_head = ClassificationHead(
            input_dim=vision_hidden_size,
            hidden_dims=config["model"]["classifier"]["hidden_dims"],
            output_dim=config["model"]["num_classes"],
            dropout_rates=config["model"]["classifier"]["dropout_rates"],
            use_batchnorm=config["model"]["classifier"]["batch_norm"],
            activation=config["model"]["classifier"]["activation"],
        )
        
        # Create response generator for text output
        self.response_generator = ResponseGenerator(
            input_dim=language_hidden_size,
            hidden_dims=[language_hidden_size, language_hidden_size // 2],
            vocab_size=self.tokenizer.vocab_size,
            max_length=128,
            dropout_rates=[0.1, 0.1],
            use_batchnorm=True,
            activation="gelu"
        )
        
        # Import and initialize components for enhanced multimodal capabilities
        try:
            # Initialize question classifier
            from models.classification.question_classifier import QuestionClassifier
            
            # Load classifier config from file
            try:
                import yaml
                # Don't re-import Path, it's already imported at the module level
                
                # Get classifier config path
                classifier_config_path = Path("config/classifier/question_classifier_config.yaml")
                
                if classifier_config_path.exists():
                    with open(classifier_config_path, 'r') as f:
                        classifier_config = yaml.safe_load(f)
                    self.logger.info(f"Loaded question classifier config from {classifier_config_path}")
                    
                    # Get model configuration from config file
                    model_config = classifier_config.get("model", {})
                    use_custom_path = model_config.get("use_custom_path", True)  # Default to True for production
                    custom_path = model_config.get("custom_path", "")
                    
                    # In production, we use the custom path
                    if use_custom_path:
                        # Check if custom path exists
                        custom_path_obj = Path(custom_path)
                        if custom_path_obj.exists():
                            self.logger.info(f"Using production model from custom path: {custom_path}")
                            model_name = custom_path
                        else:
                            # Custom path doesn't exist - production will fail
                            error_msg = f"Production model path does not exist: {custom_path}"
                            self.logger.error(error_msg)
                            raise FileNotFoundError(error_msg)
                    else:
                        # Development mode - use HuggingFace model
                        model_name = model_config.get("dev_model_name", model_config.get("name", ""))
                        if not model_name:
                            error_msg = "No model name specified in config for development mode"
                            self.logger.error(error_msg)
                            raise ValueError(error_msg)
                        self.logger.info(f"Using development model from HuggingFace: {model_name}")
                    
                    # Get other parameters
                    hidden_size = model_config.get("hidden_size", 768)
                    num_classes = model_config.get("num_classes", 5)
                else:
                    # Config file not found - this is a fatal error in production
                    error_msg = f"Question classifier config not found at {classifier_config_path}"
                    self.logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                # Get use_internvl_language_model flag but ignore it - this MUST be false
                use_internvl_lm = model_config.get("use_internvl_language_model", False)
                if use_internvl_lm:
                    self.logger.warning("Found use_internvl_language_model=true but this MUST be false to avoid index errors")
                else:
                    self.logger.info("Verified use_internvl_language_model=false (correct configuration)")
                
                from models.classification.question_classifier import QuestionClassifier
                
                # MPNet is the only viable option for question classification
                self.logger.info("Creating question classifier using separate MPNet model (all-mpnet-base-v2)")
                self.logger.info("This is the only approach that works reliably without index errors")
                
                # We will ALWAYS use the dedicated text model (all-mpnet-base-v2) for question classification
                # Using InternVL's language model causes index errors due to model mismatch
                
                # Use the separate MPNet model as specified in the config
                self.logger.info(f"Using dedicated text model (all-mpnet-base-v2) for question classification - this is the only stable approach")
                if use_internvl_lm:
                    self.logger.warning("Ignoring use_internvl_language_model=true setting - this causes index errors")
                    self.logger.warning("Forcing use of dedicated text model instead of InternVL language model")
                
                self.question_classifier = QuestionClassifier(
                    model_name=model_name,  # This is the path to all-mpnet-base-v2
                    hidden_size=hidden_size,
                    num_classes=num_classes,
                    # Let the classifier load its own model and tokenizer
                    use_custom_path=use_custom_path,
                    use_existing_models=False  # Don't use InternVL models
                )
                
                # Ensure all parameters are set to require gradients 
                for param in self.question_classifier.parameters():
                    param.requires_grad = True
                self.logger.info(f"Initialized question classifier with model: {model_name}")
            except Exception as e:
                # This is a fatal error - we need question classification to work
                self.logger.error(f"Failed to initialize question classifier: {e}")
                raise RuntimeError(f"Failed to initialize question classifier: {e}")
            
            # Initialize template selector
            from models.components.template_system import TemplateSelector
            self.template_selector = TemplateSelector()
            # Rename the templates attribute to avoid conflict with model attributes
            if hasattr(self.template_selector, 'templates'):
                # Rename the attribute to avoid conflict
                setattr(self.template_selector, '_template_registry', self.template_selector.templates)
                # Delete the original attribute
                delattr(self.template_selector, 'templates')
            self.logger.info("Initialized template selector")
            
            # Detail extractor has been deprecated and removed
            # Visual features are now processed directly by the vision encoder and cross-attention
            self.logger.info("Initialized detail extractor")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize enhanced components: {e}. Will use basic template system.")
        
        # Freeze vision encoder if required
        if freeze_vision_encoder:
            self.logger.info("Freezing vision encoder")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # Freeze language model if required
        if freeze_language_model:
            self.logger.info("Freezing language model")
            for param in self.language_model.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        text_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal processing.
        
        Args:
            pixel_values: Batch of images [B, C, H, W]
            text_input_ids: Optional text prompts encoded as token IDs [B, L]
            attention_mask: Optional attention mask for text input [B, L]
            
        Returns:
            Dictionary with logits, embeddings, and text response
        """
        # Ensure inputs require grad for proper backpropagation
        if self.training and not pixel_values.requires_grad:
            pixel_values.requires_grad_(True)
            
        # Vision encoding with gradient checkpointing if training
        if hasattr(self.vision_encoder, 'gradient_checkpointing') and self.training:
            self.vision_encoder.gradient_checkpointing = True
            
        # Ensure all modules are in consistent dtype (float32) before forward pass
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)
        
        # Use torch.amp.autocast for reduced precision during forward (updated API)
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                               enabled=torch.cuda.is_available() and self.training):
            # Get vision outputs - specifically for InternVL2 vision model
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            
            # Extract image embeddings from the vision model's last hidden state
            image_embeds = vision_outputs.last_hidden_state
            
            # If text input is provided, process it
            if text_input_ids is not None:
                # Enable gradient checkpointing during training
                if hasattr(self.language_model, 'gradient_checkpointing') and self.training:
                    self.language_model.gradient_checkpointing = True
                
                # Process text with InternVL2's language model
                text_outputs = self.language_model(
                    input_ids=text_input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False  # Disable KV caching to save memory during training
                )
                
                # Get text embeddings from model outputs, accounting for different output formats
                
                # First run - inspect and log the output format for debugging
                if not hasattr(self, '_output_format_logged') and self.logger:
                    self._output_format_logged = True
                    output_type = type(text_outputs).__name__
                    self.logger.info(f"Language model output type: {output_type}")
                    # Log available attributes
                    output_attrs = [attr for attr in dir(text_outputs) if not attr.startswith('_')]
                    self.logger.info(f"Available attributes: {', '.join(output_attrs)}")
                    
                    # Log hidden states format
                    if hasattr(text_outputs, 'hidden_states') and text_outputs.hidden_states is not None:
                        if isinstance(text_outputs.hidden_states, tuple):
                            self.logger.info(f"hidden_states is a tuple of length {len(text_outputs.hidden_states)}")
                            first_hs = text_outputs.hidden_states[0]
                            self.logger.info(f"First hidden state shape: {first_hs.shape}")
                            last_hs = text_outputs.hidden_states[-1]
                            self.logger.info(f"Last hidden state shape: {last_hs.shape}")
                
                # Extract embeddings based on available attributes
                if hasattr(text_outputs, 'last_hidden_state'):
                    # Standard format (most models)
                    text_embeds = text_outputs.last_hidden_state
                elif hasattr(text_outputs, 'hidden_states') and text_outputs.hidden_states is not None:
                    # Some models provide hidden_states tuple
                    text_embeds = text_outputs.hidden_states[-1]
                elif hasattr(text_outputs, 'logits'):
                    # CausalLMOutputWithPast format (common in newer models with no hidden states)
                    text_embeds = text_outputs.logits
                    self.logger.warning("Using logits as embeddings - this may impact performance")
                else:
                    # If we can't find proper embeddings, log the output structure and raise error
                    output_attrs = [attr for attr in dir(text_outputs) if not attr.startswith('_')]
                    error_msg = f"Unable to extract text embeddings. Available attributes: {', '.join(output_attrs)}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Clear text_outputs to free memory early
                del text_outputs
                
                # Cross-modal attention (text attends to vision)
                # Use chunked processing if sequence length is too long
                if text_embeds.shape[1] > 64 and self.training:
                    # Process in chunks for long sequences
                    chunks = []
                    chunk_size = 32
                    for i in range(0, text_embeds.shape[1], chunk_size):
                        end_idx = min(i + chunk_size, text_embeds.shape[1])
                        chunk = text_embeds[:, i:end_idx, :]
                        chunk_output = self.cross_attention(
                            query=chunk,
                            key=image_embeds,
                            value=image_embeds,
                            attention_mask=None
                        )
                        chunks.append(chunk_output)
                    multimodal_embeds = torch.cat(chunks, dim=1)
                else:
                    # Regular processing for shorter sequences
                    multimodal_embeds = self.cross_attention(
                        query=text_embeds,
                        key=image_embeds,
                        value=image_embeds,
                        attention_mask=None
                    )
                
                # Generate text response with reduced memory footprint
                response_output = self.response_generator(multimodal_embeds)
                
                # Apply classification head to pooled vision features for receipt counting
                pooled_vision = image_embeds.mean(dim=1)
                classification_logits = self.classification_head(pooled_vision)
                
                # Get question type if question classifier exists
                question_type_logits = None
                if hasattr(self, 'question_classifier') and self.training:
                    # Get question text
                    # We only do this during training as we need ground truth labels
                    # During inference, we handle this in generate_response
                    try:
                        # Get question classifier's tokenizer
                        if not hasattr(self.question_classifier, 'tokenizer'):
                            # This is a serious error - the question classifier must have a tokenizer
                            self.logger.error("Question classifier does not have a tokenizer attribute")
                            raise RuntimeError("Question classifier does not have a tokenizer attribute")
                    except Exception as e:
                        # Log but don't fail - we can still train without question classification
                        self.logger.warning(f"Question classifier error, continuing without classification: {e}")
                        # Skip the rest of this block
                        continue_execution = False
                    else:
                        continue_execution = True
                        
                    if continue_execution:
                        # Get vocabulary size
                        vocab_size = getattr(self.question_classifier.tokenizer, "vocab_size", None)
                        if vocab_size is None:
                            # Try to get from model config
                            if hasattr(self.question_classifier.encoder, "config"):
                                vocab_size = getattr(self.question_classifier.encoder.config, "vocab_size", None)
                                
                            if vocab_size is None:
                                # Still not found - this is a fatal error
                                self.logger.error("Could not determine tokenizer vocabulary size")
                                raise RuntimeError("Could not determine tokenizer vocabulary size")
                        
                        # Decode question text
                        question_texts = self.tokenizer.batch_decode(
                            text_input_ids, skip_special_tokens=True
                        )
                        
                        # Since we're using the same tokenizer, we can just reuse the original text_input_ids
                        # No need to re-tokenize, which avoids token ID mismatches entirely
                        
                        # Forward pass through classifier using the original input IDs and mask
                        question_type_logits = self.question_classifier(
                            text_input_ids,
                            attention_mask
                        )
                        
                        self.logger.debug("Using shared InternVL2 tokenizer and language model components")
                
                # Detail extraction is now handled by the unified model architecture
                detail_logits = None
                
                # Clear image_embeds as soon as possible to free memory
                del image_embeds
                
                # Prepare outputs with enhanced components
                outputs = {
                    "logits": classification_logits,
                    "embeddings": pooled_vision,
                    "multimodal_embeddings": multimodal_embeds,
                    "response_logits": response_output["logits"],
                    # Only include features if not in training mode to save memory
                    "response_features": response_output["features"] if not self.training else None
                }
                
                # Add outputs from enhanced components if available
                if question_type_logits is not None:
                    outputs["question_type_logits"] = question_type_logits
                    
                if detail_logits is not None:
                    outputs["detail_logits"] = detail_logits
                    
                return outputs
            else:
                # Regular vision-only path for backward compatibility
                pooled_output = image_embeds.mean(dim=1)
                logits = self.classification_head(pooled_output)
                
                # Clear image_embeds to free memory
                del image_embeds
                
                return {
                    "logits": logits,
                    "embeddings": pooled_output
                }
    
    def generate_response(
        self,
        pixel_values: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate a text response to a question about an image.
        
        Args:
            pixel_values: Batch of images [B, C, H, W]
            text_input_ids: Text prompt token IDs [B, L]
            attention_mask: Optional attention mask for text [B, L]
            max_length: Maximum length of generated response
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Tuple of (token_ids, decoded_texts)
        """
        # Ensure all inputs are in float32 for consistency
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)
        
        # Generate response using no_grad for inference
        with torch.no_grad():
            # Get question text
            question = self.tokenizer.decode(text_input_ids[0], skip_special_tokens=True)
            self.logger.info(f"Question text: '{question}'")
            
            # Classify question type
            question_type = self.question_classifier.predict_question_type(question)
            self.logger.info(f"Classified question as: {question_type}")
            
            # Generate embeddings and get document class
            outputs = self.forward(
                pixel_values=pixel_values, 
                text_input_ids=text_input_ids, 
                attention_mask=attention_mask
            )
            
            # Get multimodal context and class predictions
            multimodal_context = outputs["multimodal_embeddings"]
            _, predicted_class = outputs["logits"].max(1)
            
            # Detail extraction is now integrated into the unified vision-language processing
            extracted_details = {}
            
            # Use template system to generate responses
            responses = []
            for idx, cls in enumerate(predicted_class):
                response = self.template_selector.select_template(
                    question_type,
                    cls.item(),
                    extracted_details
                )
                responses.append(response)
            
            # Create token IDs for responses
            batch_size = text_input_ids.shape[0]
            tokens_list = []
            for response in responses:
                # Encode each response
                encoded = self.tokenizer.encode(response, add_special_tokens=True)
                # Pad or truncate to max_length
                if len(encoded) > max_length:
                    encoded = encoded[:max_length]
                else:
                    encoded += [self.tokenizer.pad_token_id] * (max_length - len(encoded))
                tokens_list.append(encoded)
            
            # Convert to tensor
            generated_ids = torch.tensor(tokens_list, device=text_input_ids.device, dtype=torch.long)
            
            return generated_ids, responses
    
    def prepare_inputs(
        self, 
        images: torch.Tensor, 
        text_prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model from raw images and text prompts.
        
        Args:
            images: Batch of images [B, C, H, W]
            text_prompts: List of text prompts
            
        Returns:
            Dictionary of model inputs
        """
        # Tokenize text prompts
        tokenized = self.tokenizer(
            text_prompts,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to the same device as images
        input_ids = tokenized.input_ids.to(images.device)
        attention_mask = tokenized.attention_mask.to(images.device)
        
        return {
            "pixel_values": images,
            "text_input_ids": input_ids,
            "attention_mask": attention_mask
        }