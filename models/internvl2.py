"""
InternVL2 model implementation for receipt counting with vision-language integration.

This module adapts the InternVL2 vision-language model for:
1. Receipt counting classification using vision encoder
2. Multimodal vision-language processing to answer queries about receipts
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

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
            raise ValueError(f"Model path does not exist: {pretrained_path}. Please provide a valid path to the pre-downloaded model.")
        
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
                    self.logger.warning("Flash Attention requested but not available. Install with: CUDA_HOME=/usr/local/cuda pip install flash-attn>=2.5.0")
            
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
        
        # Extract vision encoder from the full model
        if hasattr(self.model, "vision_model"):
            self.vision_encoder = self.model.vision_model
            self.logger.info("Using vision_model for vision encoding")
        else:
            # For newer versions/implementation, vision_model might be accessed differently
            try:
                self.vision_encoder = self.model.vision_encoder
                self.logger.info("Using vision_encoder for vision encoding")
            except:
                try:
                    self.vision_encoder = self.model.vision_tower
                    self.logger.info("Using vision_tower for vision encoding")
                except:
                    self.logger.warning("Could not extract vision encoder directly. Using full model.")
                    self.vision_encoder = self.model  # Fallback to using full model
        
        # Specifically disable sliding window attention and other problematic settings
        try:
            # Check for config access points
            if hasattr(self.model, "config"):
                if hasattr(self.model.config, "use_sliding_window"):
                    self.model.config.use_sliding_window = False
                    self.logger.info("Disabled sliding window attention in model config")
                    
            if hasattr(self.vision_encoder, "config"):
                if hasattr(self.vision_encoder.config, "use_sliding_window"):
                    self.vision_encoder.config.use_sliding_window = False
                    self.logger.info("Disabled sliding window attention in vision encoder config")
        except Exception as e:
            self.logger.warning(f"Could not disable sliding window attention: {e}")
        
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
            {'params': [p for p in self.vision_encoder.parameters() if p.requires_grad], 'lr': lr_multiplier}  # Reduced LR
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
        
        # Pass through vision encoder
        try:
            # Normal execution path (no forced error)
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            
            # Try different output structures
            if hasattr(vision_outputs, 'last_hidden_state'):
                image_embeds = vision_outputs.last_hidden_state
            elif hasattr(vision_outputs, 'hidden_states'):
                image_embeds = vision_outputs.hidden_states[-1]  # Use the last layer
            elif isinstance(vision_outputs, tuple) and len(vision_outputs) > 0:
                image_embeds = vision_outputs[0]  # First element is often the hidden states
            else:
                # Assume the output is already the embeddings
                image_embeds = vision_outputs
                
        except Exception as e:
            self.logger.warning(f"Error in forward pass: {e}")
            # Create a simple vision encoder using a ResNet backbone
            try:
                # Try with direct model call first
                vision_outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
                if hasattr(vision_outputs, 'vision_model_output'):
                    image_embeds = vision_outputs.vision_model_output.last_hidden_state
                elif hasattr(vision_outputs, 'hidden_states'):
                    image_embeds = vision_outputs.hidden_states[-1]
                else:
                    raise ValueError(f"Could not extract image embeddings from model outputs: {type(vision_outputs)}")
            except Exception as e2:
                self.logger.error(f"Second attempt failed with error: {e2}. Using a simplified vision encoder for testing.")
                
                # Create a simple CNN feature extractor as fallback
                if not hasattr(self, '_fallback_encoder'):
                    self.logger.info("Creating fallback CNN encoder")
                    import torchvision.models as models
                    
                    # Load a pre-trained ResNet model
                    resnet = models.resnet18(weights="DEFAULT")
                    # Remove the final fully connected layer
                    self._fallback_encoder = torch.nn.Sequential(
                        *[module for i, module in enumerate(resnet.children()) if i < 8]
                    )
                    # Freeze parameters
                    for param in self._fallback_encoder.parameters():
                        param.requires_grad = False
                    # Convert to the right dtype
                    self._fallback_encoder = self._fallback_encoder.to(pixel_values.dtype).to(pixel_values.device)
                    
                # Get features from the fallback encoder
                features = self._fallback_encoder(pixel_values)
                
                # Reshape to match expected format (B, seq_len, hidden_dim)
                batch_size = pixel_values.shape[0]
                features = features.permute(0, 2, 3, 1)  # B, H, W, C
                features = features.reshape(batch_size, -1, features.shape[-1])  # B, H*W, C
                
                # Limit sequence length to 256 if needed
                if features.shape[1] > 256:
                    features = features[:, :256, :]
                
                # Pad to 256 tokens if needed
                if features.shape[1] < 256:
                    pad_size = 256 - features.shape[1]
                    padding = torch.zeros(batch_size, pad_size, features.shape[-1], 
                                         dtype=features.dtype, device=features.device)
                    features = torch.cat([features, padding], dim=1)
                
                image_embeds = features
        
        # Global average pooling over sequence dimension
        pooled_output = image_embeds.mean(dim=1)
        
        # Simplified dtype handling - we know the types match from initialization
        # This section kept in case initialization fails or model is modified at runtime
        if not hasattr(self, '_model_classifier_dtypes_matched'):
            if hasattr(self.classification_head, 'mlp') and len(self.classification_head.mlp) > 0:
                if hasattr(self.classification_head.mlp[0], 'weight'):
                    target_dtype = self.classification_head.mlp[0].weight.dtype
                    if pooled_output.dtype != target_dtype:
                        self.logger.info(f"One-time conversion from {pooled_output.dtype} to {target_dtype}")
                        self._model_classifier_dtypes_matched = True
                    else:
                        self.logger.info(f"No conversion needed: model output already {pooled_output.dtype}")
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
                    if isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
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
            raise ValueError(f"Model path does not exist: {pretrained_path}. Please provide a valid path to the pre-downloaded model.")
        
        # Load the full model with vision and language components
        if pretrained:
            self.logger.info(f"Loading model from local path: {pretrained_path}")
            # Set up the right parameters based on config
            kwargs = {
                # "device_map": "auto",  # Commented for macOS development - re-enable for Linux GPU
                "trust_remote_code": True,
                "local_files_only": True,
                # "max_memory": {0: "16GB"},  # Commented for macOS development - re-enable for Linux GPU
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
        
        # Extract vision and language components
        if hasattr(self.model, "vision_model"):
            self.vision_encoder = self.model.vision_model
            self.logger.info("Using vision_model for vision encoding")
        else:
            try:
                self.vision_encoder = self.model.vision_encoder
                self.logger.info("Using vision_encoder for vision encoding")
            except:
                try:
                    self.vision_encoder = self.model.vision_tower
                    self.logger.info("Using vision_tower for vision encoding")
                except:
                    self.logger.warning("Could not extract vision encoder directly. Using full model.")
                    self.vision_encoder = self.model
        
        # Get the language model component
        if hasattr(self.model, "language_model"):
            self.language_model = self.model.language_model
            self.logger.info("Using language_model for text encoding")
        else:
            try:
                self.language_model = self.model.text_encoder
                self.logger.info("Using text_encoder for text encoding")
            except:
                try:
                    self.language_model = self.model.text_model
                    self.logger.info("Using text_model for text encoding")
                except:
                    # If language model component was deleted, restore it
                    self.logger.warning("Language model not found. Loading a new language model instance.")
                    try:
                        # Try to get a compatible language model from the hub
                        from transformers import AutoModelForCausalLM
                        self.language_model = AutoModelForCausalLM.from_pretrained(
                            pretrained_path,
                            trust_remote_code=True,
                            local_files_only=True,
                            # device_map="auto"  # Commented for macOS development - re-enable for Linux GPU
                        )
                        self.logger.info("Loaded language model successfully.")
                    except Exception as e:
                        self.logger.error(f"Error loading language model: {e}")
                        raise ValueError("Could not instantiate a language model. Please ensure the model has a language component.")
        
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
        # Vision encoding
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        
        # Extract image embeddings
        if hasattr(vision_outputs, 'last_hidden_state'):
            image_embeds = vision_outputs.last_hidden_state
        elif hasattr(vision_outputs, 'hidden_states'):
            image_embeds = vision_outputs.hidden_states[-1]
        elif isinstance(vision_outputs, tuple) and len(vision_outputs) > 0:
            image_embeds = vision_outputs[0]
        else:
            image_embeds = vision_outputs
        
        # If text input is provided, process it
        if text_input_ids is not None:
            # Text encoding
            text_outputs = self.language_model(
                input_ids=text_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract text embeddings
            if hasattr(text_outputs, 'last_hidden_state'):
                text_embeds = text_outputs.last_hidden_state
            elif hasattr(text_outputs, 'hidden_states'):
                text_embeds = text_outputs.hidden_states[-1]
            else:
                # Get the last element from hidden states tuple
                text_embeds = text_outputs['hidden_states'][-1]
            
            # Cross-modal attention (text attends to vision)
            multimodal_embeds = self.cross_attention(
                query=text_embeds, 
                key=image_embeds, 
                value=image_embeds,
                attention_mask=None  # No specific cross-attention mask in this implementation
            )
            
            # Generate text response
            response_output = self.response_generator(multimodal_embeds)
            
            # Apply classification head to pooled vision features for receipt counting
            pooled_vision = image_embeds.mean(dim=1)
            classification_logits = self.classification_head(pooled_vision)
            
            return {
                "logits": classification_logits,
                "embeddings": pooled_vision,
                "multimodal_embeddings": multimodal_embeds,
                "response_logits": response_output["logits"],
                "response_features": response_output["features"]
            }
        else:
            # Regular vision-only path for backward compatibility
            pooled_output = image_embeds.mean(dim=1)
            logits = self.classification_head(pooled_output)
            
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
        # Run forward pass to get multimodal embeddings
        outputs = self.forward(
            pixel_values=pixel_values, 
            text_input_ids=text_input_ids, 
            attention_mask=attention_mask
        )
        
        # Get multimodal context for generation
        multimodal_context = outputs["multimodal_embeddings"]
        
        # Use response generator to create response
        # Starting with BOS token or the last token from the prompt
        batch_size = text_input_ids.shape[0]
        
        # Get start tokens (using the last token of the input as context)
        start_tokens = text_input_ids[:, -1:].clone()
        
        # Generate response tokens
        generated_ids = self.response_generator.generate(
            start_tokens=start_tokens,
            multimodal_context=multimodal_context,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Decode the token IDs to text
        decoded_texts = []
        for ids in generated_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            decoded_texts.append(text)
        
        return generated_ids, decoded_texts
    
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