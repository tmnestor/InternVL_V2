# Migration Plan: Transitioning from InternVL2_5-1B to Llama-3.2-11B-Vision

## Overview

This document outlines the step-by-step migration plan to transition the InternVL_V2 system from using InternVL2_5-1B to meta-llama/Llama-3.2-11B-Vision. The migration will enhance the system's vision-language capabilities while requiring architectural and configuration changes to accommodate the different model architecture.

## 1. Model Architecture Differences

### InternVL2_5-1B
- Architecture: Based on InternVLForCausalLM / InternVLChatModel
- Vision Encoder: Specialized InternVL vision transformer
- Language Model: Proprietary language model
- Parameters: 5B parameters
- Integration: Custom cross-attention mechanism

### Llama-3.2-11B-Vision
- Architecture: Based on LlamaForCausalLM with vision projector 
- Vision Encoder: CLIP ViT vision encoder
- Language Model: Llama 3.2 decoder-only architecture
- Parameters: 11B parameters
- Integration: Vision tokens projected directly into language model's embedding space

## 2. Migration Steps

### 2.1. Update Model Configuration

```yaml
# Updated configuration for Llama-3.2-11B-Vision
model:
  # New model path
  pretrained_path: "/path/to/meta-llama/Llama-3.2-11B-Vision"
  # Set appropriate model type
  model_type: "llama-vision"
  multimodal: true
  num_classes: 3
  use_8bit: true  # Enable 8-bit quantization to manage memory usage for 11B model
  
  # Classification head configuration (remains similar)
  classifier:
    hidden_dims: [768, 256]  # Updated to match Llama's hidden dimensions
    dropout_rates: [0.2, 0.1]
    batch_norm: true
    activation: "gelu"
```

### 2.2. Model Adapter Code Changes

Create a new adapter for Llama-3.2-11B-Vision in `models/vision_language/llama_vision.py`:

```python
"""
Llama-3.2-11B-Vision model implementation for receipt counting with vision-language integration.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    CLIPVisionModel
)

from models.components.projection_head import ClassificationHead, ResponseGenerator


class LlamaVisionReceiptClassifier(nn.Module):
    """
    Llama-3.2-11B-Vision-based receipt classification model.
    
    Adapts the Llama-3.2-11B-Vision architecture for the receipt counting task
    by using the vision encoder and adding a custom classification head.
    """
    def __init__(
        self, 
        config: Dict[str, Any],
        pretrained: bool = True,
        freeze_vision_encoder: bool = False,
    ):
        """
        Initialize the Llama Vision receipt classifier.
        
        Args:
            config: Model configuration
            pretrained: Whether to load pretrained weights
            freeze_vision_encoder: Whether to freeze the vision encoder
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the Llama Vision model
        pretrained_path = config["model"]["pretrained_path"]
        use_8bit = config["model"].get("use_8bit", False)
        
        # Verify the path exists
        if not Path(pretrained_path).exists():
            raise ValueError(
                f"Model path does not exist: {pretrained_path}. Please provide a valid path to the model."
            )
        
        # Load model with appropriate settings
        if pretrained:
            self.logger.info(f"Loading model from local path: {pretrained_path}")
            kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
            }
            
            # Enable 8-bit quantization if specified (recommended for 11B model)
            if use_8bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_skip_modules=["vision_model", "vision_tower"]
                    )
                    kwargs["quantization_config"] = quantization_config
                    self.logger.info("Using 8-bit quantization for memory efficiency")
                except ImportError:
                    self.logger.warning("BitsAndBytesConfig not available, using default precision")
            
            # Load the full model
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_path,
                **kwargs
            )
            
            # Get tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        else:
            # Load with default initialization (for debugging)
            config_obj = AutoConfig.from_pretrained(
                pretrained_path, trust_remote_code=True, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_config(config_obj)
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_path, trust_remote_code=True, local_files_only=True
            )
        
        # Extract vision encoder from the Llama model
        # Note: Llama-3.2-11B-Vision has vision_tower attribute for vision encoding
        self.vision_encoder = self.model.get_vision_tower()
        self.logger.info("Using vision_tower for vision encoding")
        
        # Get vision encoder output dimension
        vision_hidden_size = self.vision_encoder.config.hidden_size
        self.logger.info(f"Vision encoder hidden size: {vision_hidden_size}")
        
        # Create vision projection layer (Llama models require this)
        self.vision_projection = nn.Linear(
            vision_hidden_size,
            self.model.config.hidden_size
        )
        
        # Create a custom classification head
        self.classification_head = ClassificationHead(
            input_dim=self.model.config.hidden_size,  # Use Llama's hidden size
            hidden_dims=config["model"]["classifier"]["hidden_dims"],
            output_dim=config["model"]["num_classes"],
            dropout_rates=config["model"]["classifier"]["dropout_rates"],
            use_batchnorm=config["model"]["classifier"]["batch_norm"],
            activation=config["model"]["classifier"]["activation"],
        )
        
        # Freeze vision encoder if required
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            pixel_values: Batch of images [B, C, H, W]
            
        Returns:
            Dictionary with logits and other outputs
        """
        # Process images through vision encoder
        with torch.no_grad() if not self.training else torch.enable_grad():
            vision_outputs = self.vision_encoder(pixel_values)
            
            # Get visual embeddings - Llama vision models typically use pooled output
            if hasattr(vision_outputs, "pooler_output"):
                image_embeds = vision_outputs.pooler_output
            else:
                # Fallback to mean pooling of last hidden state
                image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
        
        # Project vision embeddings to language model dimension
        projected_embeddings = self.vision_projection(image_embeds)
        
        # Pass through classifier head
        logits = self.classification_head(projected_embeddings)
        
        return {
            "logits": logits,
            "embeddings": projected_embeddings
        }
```

### 2.3. Update MultimodalModel Implementation

Create a new multimodal implementation for Llama-3.2-11B-Vision in the same file:

```python
class LlamaVisionMultimodalModel(nn.Module):
    """
    Llama-3.2-11B-Vision-based multimodal model for vision-language tasks.
    
    Adapts the Llama Vision architecture to handle both vision and language inputs,
    for answering questions about receipt images.
    """
    def __init__(
        self, 
        config: Dict[str, Any],
        pretrained: bool = True,
        freeze_vision_encoder: bool = True,
        freeze_language_model: bool = False,
    ):
        """
        Initialize the Llama Vision multimodal model.
        
        Args:
            config: Model configuration
            pretrained: Whether to load pretrained weights
            freeze_vision_encoder: Whether to freeze the vision encoder
            freeze_language_model: Whether to freeze the language model
        """
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize the Llama Vision model with appropriate settings
        pretrained_path = config["model"]["pretrained_path"]
        use_8bit = config["model"].get("use_8bit", False)
        
        # Verify the path exists
        if not Path(pretrained_path).exists():
            raise ValueError(
                f"Model path does not exist: {pretrained_path}. Please provide a valid path to the model."
            )
        
        # Load the Llama Vision model
        if pretrained:
            self.logger.info(f"Loading model from local path: {pretrained_path}")
            kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
            }
            
            # Enable 8-bit quantization if specified (recommended for 11B model)
            if use_8bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_skip_modules=["vision_model", "vision_tower"]
                    )
                    kwargs["quantization_config"] = quantization_config
                    self.logger.info("Using 8-bit quantization for memory efficiency")
                except ImportError:
                    self.logger.warning("BitsAndBytesConfig not available, using default precision")
            
            # Load the model
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_path,
                **kwargs
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        else:
            # Load with default initialization (for debugging)
            config_obj = AutoConfig.from_pretrained(
                pretrained_path, trust_remote_code=True, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_config(config_obj)
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_path, trust_remote_code=True, local_files_only=True
            )
        
        # Extract components
        self.vision_encoder = self.model.get_vision_tower()
        self.language_model = self.model  # Full model is needed for language processing
        
        # Get hidden sizes for both encoders
        vision_hidden_size = self.vision_encoder.config.hidden_size
        language_hidden_size = self.model.config.hidden_size
        
        # Create vision projection layer
        self.vision_projection = nn.Linear(
            vision_hidden_size,
            language_hidden_size
        )
        
        # Create a custom classification head
        self.classification_head = ClassificationHead(
            input_dim=language_hidden_size,
            hidden_dims=config["model"]["classifier"]["hidden_dims"],
            output_dim=config["model"]["num_classes"],
            dropout_rates=config["model"]["classifier"]["dropout_rates"],
            use_batchnorm=config["model"]["classifier"]["batch_norm"],
            activation=config["model"]["classifier"]["activation"],
        )
        
        # Create response generator (simpler for Llama)
        self.response_generator = nn.Linear(
            language_hidden_size,
            self.tokenizer.vocab_size
        )
        
        # Freeze vision encoder if required
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # Freeze language model if required
        if freeze_language_model:
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
        # Process images through vision encoder
        with torch.no_grad() if not self.training else torch.enable_grad():
            vision_outputs = self.vision_encoder(pixel_values)
            
            # Get visual embeddings
            if hasattr(vision_outputs, "pooler_output"):
                image_embeds = vision_outputs.pooler_output
            else:
                # Fallback to mean pooling of last hidden state
                image_embeds = vision_outputs.last_hidden_state.mean(dim=1)
        
        # Project vision embeddings to language model dimension
        projected_image_embeds = self.vision_projection(image_embeds)
        
        if text_input_ids is not None:
            # Format prompts with image embeddings for Llama-Vision
            # This is a simplified approach - in production, use the proper formulation
            # with vision tokens embedded in the right positions
            
            # Use model's native image handling mechanism as documented
            model_inputs = self.model.prepare_inputs_for_generation(
                text_input_ids,
                attention_mask=attention_mask,
                images=pixel_values  # Pass images directly - model will handle them
            )
            
            # Forward pass through model with proper inputs
            outputs = self.model(**model_inputs, return_dict=True)
            
            # Get language model hidden states
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs.last_hidden_state
            
            # Apply classification on the projected image embeddings
            classification_logits = self.classification_head(projected_image_embeds)
            
            # Generate text logits from response generator
            # Use the [0] position as many models use this for generation start
            response_logits = self.response_generator(hidden_states)
            
            # Return all outputs
            return {
                "logits": classification_logits,
                "embeddings": projected_image_embeds,
                "response_logits": response_logits,
                "multimodal_embeddings": hidden_states
            }
        else:
            # Image-only mode (classification)
            classification_logits = self.classification_head(projected_image_embeds)
            
            return {
                "logits": classification_logits,
                "embeddings": projected_image_embeds
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
        # Use model's native generation capabilities
        with torch.no_grad():
            # Create inputs with both image and text
            # Llama Vision models can handle this natively
            model_inputs = self.model.prepare_inputs_for_generation(
                text_input_ids,
                attention_mask=attention_mask,
                images=pixel_values  # Pass images directly
            )
            
            # Generate response tokens
            output_ids = self.model.generate(
                **model_inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode tokens to text
            decoded_texts = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            
            return output_ids, decoded_texts
```

### 2.4. Update Factory Method in `models/__init__.py`

Update the model factory to include the new Llama Vision model:

```python
def create_model(config, model_type=None):
    """Create model based on configuration or specified type."""
    if model_type is None:
        model_type = config.get("model", {}).get("model_type", "internvl")
    
    if model_type.lower() == "internvl":
        from models.vision_language.internvl2 import InternVL2ReceiptClassifier
        return InternVL2ReceiptClassifier(config)
    elif model_type.lower() == "llama-vision":
        from models.vision_language.llama_vision import LlamaVisionReceiptClassifier
        return LlamaVisionReceiptClassifier(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
```

### 2.5. Memory Optimization

Llama-3.2-11B-Vision requires significantly more memory than InternVL2_5-1B. Update the trainer configuration to manage memory usage:

```yaml
# Training configuration for Llama Vision
training:
  epochs: 15
  learning_rate: 5.0e-5
  weight_decay: 1.0e-4
  warmup_steps: 200
  gradient_accumulation_steps: 8  # Increased from 4 to reduce memory footprint
  flash_attention: true  # Enable if available
  fp16: false
  gradient_clip: 1.0
  torch_compile: false
  memory_efficient: true
  low_cpu_mem_usage: true
  bf16: true  # Use bfloat16 for better numerical stability
  
  # Reduce batch size to manage memory
  batch_size: 2  # Reduced from 4
```

### 2.6. Update Data Processing

Llama-3.2-11B-Vision uses a different tokenizer and image processing pipeline than InternVL2. Update the data loaders:

```python
def create_llama_vision_transforms():
    """Create image transformations for Llama Vision model."""
    # Llama Vision models typically use CLIP's preprocessing
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

def process_llama_text_prompts(prompts, tokenizer, max_length=128):
    """Process text prompts for Llama Vision models."""
    # Llama-3.2-11B-Vision requires specific prompt formatting
    formatted_prompts = [
        f"<image>\n{prompt}" for prompt in prompts
    ]
    
    # Tokenize with Llama's tokenizer
    return tokenizer(
        formatted_prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
```

## 3. Model Download & Access Instructions

Llama-3.2-11B-Vision requires proper access credentials:

1. Request access through Meta AI's website: https://ai.meta.com/resources/models-and-libraries/llama-downloads/

2. Download the model using the provided script:
   ```bash
   python -m llama_recipes.download --model meta-llama/Llama-3.2-11B-Vision
   ```

3. Install the required dependencies:
   ```bash
   pip install transformers==4.40.0 accelerate bitsandbytes
   ```

## 4. Usage Differences

### 4.1 Prompt Format

Llama-3.2-11B-Vision uses a specific prompt format with visual tokens:

```
<image>
[Question about the image]
```

### 4.2 Visual Token Handling

Llama-3.2-11B-Vision handles visual tokens differently than InternVL2:

- InternVL2 uses a cross-attention mechanism
- Llama Vision uses projected visual tokens injected directly into the language model

### 4.3 Text Generation

Llama-3.2-11B-Vision is a decoder-only model, which makes text generation more straightforward:

```python
# Generate text for an image-text pair
outputs = model.generate(
    pixel_values=images,
    text_input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=100
)
```

## 5. Migration Timeline

| Stage | Task | Timeline | Dependencies |
|-------|------|----------|--------------|
| **1. Preparation** | Model download and setup | Day 1 | Access credentials |
| **2. Development** | Adapter code implementation | Days 2-4 | Model availability |
| **3. Config Updates** | Configuration file changes | Day 5 | Adapter code |
| **4. Testing** | Initial testing for compatibility | Days 6-7 | All components |
| **5. Training** | Test training runs | Days 8-10 | Testing completion |
| **6. Optimization** | Performance optimization | Days 11-14 | Training results |
| **7. Deployment** | Full migration | Day 15 | All previous stages |

## 6. Rollback Plan

In case of issues with the Llama-3.2-11B-Vision migration, follow these steps to roll back:

1. Revert configuration files to use InternVL2_5-1B
2. Revert code changes related to the Llama Vision adapter
3. Restart training with the original InternVL2 settings

## 7. Compatibility Notes

### Hardware Requirements

- **InternVL2_5-1B**: ~20GB GPU memory minimum
- **Llama-3.2-11B-Vision**: ~40GB GPU memory minimum (requires 8-bit quantization)

### Performance Considerations

- **Training Time**: Llama Vision models may require longer training due to larger parameter count
- **Inference Time**: Expect approximately 1.5-2x longer inference times
- **Accuracy**: Should be higher in complex vision-language tasks due to larger model capacity

## 8. Migration Verification Checklist

- [ ] Llama-3.2-11B-Vision model loads successfully
- [ ] Image processing pipeline works correctly
- [ ] Classification head produces expected outputs
- [ ] Text generation works correctly 
- [ ] Training loop runs without memory issues
- [ ] Validation metrics show expected accuracy
- [ ] Inference speed is acceptable for production

## 9. References

1. [Llama-3.2-11B-Vision Documentation](https://ai.meta.com/resources/models-and-libraries/llama/)
2. [Meta AI Models GitHub](https://github.com/meta-llama/llama-recipes/)
3. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/llama)