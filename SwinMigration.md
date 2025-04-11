# Swin Transformer Migration Plan for InternVL2

This document outlines a detailed plan for replacing the Vision Transformer (ViT) component in InternVL2 with a Swin Transformer architecture.

## 1. Background

### Current Architecture
- InternVL2 currently uses a standard Vision Transformer (ViT) for image encoding
- The vision encoder outputs embeddings that are processed by cross-attention mechanisms
- The current model freezes the vision encoder during initial training stages

### Swin Transformer Benefits
- Hierarchical feature representation with multi-scale processing
- Linear computational complexity to image size due to local attention windows
- State-of-the-art performance on various vision tasks
- Potentially better handling of high-resolution images

## 2. Migration Process

### 2.1 Code Analysis and Preparation

1. **Identify integration points**:
   - `models/internvl2.py`: Primary location for model architecture
   - `InternVL2MultimodalModel` class: Main multimodal model implementation
   - Vision encoder initialization and forward pass sections

2. **Analyze dimension compatibility**:
   - Current ViT output dimensions: Check `vision_hidden_size` variable
   - Swin Transformer output dimensions: Depends on variant (Tiny: 768, Small: 768, Base: 1024, Large: 1536)
   - Projection layer may be needed for dimension matching

3. **Environment preparation**:
   - Ensure transformers library is version 4.28.0+ for Swin support
   - Add additional dependencies if needed

### 2.2 Implementation Steps

1. **Model Initialization Modifications**:

```python
# Replace ViT initialization with Swin
from transformers import AutoImageProcessor, SwinModel

# Initialize Swin model
swin_model_name = "microsoft/swin-base-patch4-window7-224"  # Choose appropriate variant
self.image_processor = AutoImageProcessor.from_pretrained(swin_model_name)
self.vision_encoder = SwinModel.from_pretrained(swin_model_name)

# Add projection layer if dimensions don't match
swin_output_dim = self.vision_encoder.config.hidden_size
if swin_output_dim != vision_hidden_size:
    self.vision_projection = nn.Linear(swin_output_dim, vision_hidden_size)
```

2. **Forward Pass Adaptation**:

```python
# Modify forward pass to handle Swin outputs
def forward(self, pixel_values, text_input_ids=None, attention_mask=None):
    # Vision encoding with Swin
    vision_outputs = self.vision_encoder(pixel_values=pixel_values)
    
    # Swin returns last_hidden_state differently formatted than ViT
    # The output is [batch_size, sequence_length, hidden_size]
    image_embeds = vision_outputs.last_hidden_state
    
    # Apply projection if dimensions don't match
    if hasattr(self, 'vision_projection'):
        image_embeds = self.vision_projection(image_embeds)
    
    # Continue with existing cross-attention logic...
```

3. **Unfreezing Logic Update**:

```python
def _unfreeze_vision_encoder(self, lr_multiplier=0.01):
    """
    Update unfreezing logic to match Swin's architecture which has:
    - patch_embed layers
    - layers (list of stage modules)
    - norm layer
    """
    # Start by freezing all vision encoder parameters
    for param in self.vision_encoder.parameters():
        param.requires_grad = False
    
    # Selectively unfreeze the last stage and norm layers
    for name, param in self.vision_encoder.named_parameters():
        if "layers.3" in name or "norm" in name:  # Last stage (index 3) and normalization
            param.requires_grad = True
            
    # Include projection layer if it exists
    if hasattr(self, 'vision_projection'):
        for param in self.vision_projection.parameters():
            param.requires_grad = True
    
    # Return parameters for optimizer config
    return [p for p in self.vision_encoder.parameters() if p.requires_grad]
```

### 2.3 Training Configuration Updates

Update `config/multimodal_config.yaml`:

```yaml
model:
  pretrained_path: "/Users/tod/PretrainedLLM/InternVL2_5-1B" 
  vision_model: "microsoft/swin-base-patch4-window7-224"  # Add this line
  vision_projection_dim: 512  # Add this if needed for dimension matching
  multimodal: true
  num_classes: 3
```

### 2.4 Input Processing Adjustments

Update image preprocessing in `data/dataset.py`:

1. Ensure image sizes match Swin Transformer requirements:
   - Swin-Tiny/Small/Base: 224x224
   - Swin-Large: 384x384

```python
# Update transforms to match Swin requirements
if config["model"].get("vision_model", "").startswith("microsoft/swin"):
    if "large" in config["model"]["vision_model"]:
        image_size = 384
    else:
        image_size = 224
else:
    image_size = config["data"]["image_size"]
```

## 3. Testing and Validation

1. **Unit Tests**:
   - Test vision encoder output shapes
   - Verify cross-attention compatibility
   - Test end-to-end forward pass

2. **Integration Tests**:
   - Verify training loop works with new architecture
   - Test multi-stage training with unfreezing

3. **Performance Evaluation**:
   - Compare accuracy metrics with original ViT model
   - Measure inference speed and memory usage

## 4. Optimization Opportunities

1. **Hierarchical Feature Utilization**:
   - Consider using features from multiple resolution stages instead of just the final output
   - Implement feature pyramid networks for multi-scale feature integration

2. **Attention Window Size Tuning**:
   - Experiment with different window sizes for the Swin Transformer
   - Default is 7x7, but could be adjusted for document images like receipts

3. **Mixed Precision Training**:
   - Enable FP16 precision for faster training with Swin Transformer

## 5. Fallback Plan

1. Keep original ViT implementation available with a config flag
2. Implement graceful fallback if Swin initialization fails
3. Create adapter classes that can wrap either architecture with a common interface

## 6. Timeline

1. **Phase 1**: Architecture analysis and code preparation (1 day)
2. **Phase 2**: Implementation of Swin integration (2 days)
3. **Phase 3**: Testing and debugging (1-2 days)
4. **Phase 4**: Performance optimization (1-2 days)
5. **Phase 5**: Documentation and final review (1 day)

## 7. References

1. [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
2. [Hugging Face Swin Documentation](https://huggingface.co/docs/transformers/model_doc/swin)
3. [Microsoft Swin GitHub](https://github.com/microsoft/Swin-Transformer)