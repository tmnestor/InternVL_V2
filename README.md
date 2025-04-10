# InternVL2 Receipt Counter

A vision-language multimodal system that helps taxation officers accurately count, analyze, and extract information from receipt images using natural language interaction.

## Overview

InternVL2 Receipt Counter is built on the powerful InternVL2-5-1B model, combining advanced vision processing with natural language capabilities. It can:

- Count receipts in images with high accuracy
- Extract key information from receipts (values, dates, vendors)
- Answer natural language questions about receipts
- Generate accurate text responses based on visual content

This project implements the complete pipeline for training, evaluating, and deploying a multimodal vision-language model specifically designed for receipt analysis tasks.

## Key Features

### Multimodal Capabilities
- Integrated vision and language processing
- Natural language interaction with visual receipt data
- Cross-modal attention between text queries and images
- Contextual text generation based on visual content

### Advanced Model Architecture
- Based on InternVL2-5-1B (1 billion parameter model)
- 448×448 high-resolution image processing
- Three-stage training strategy for optimal performance
- Custom cross-attention mechanisms for vision-language fusion

### Performance Optimization
- Mixed precision training with BFloat16 support
- Flash Attention 2 for efficient transformer operations
- Memory optimization with 8-bit quantization
- Multi-GPU support with DeepSpeed and PyTorch DDP

### Development Features
- Comprehensive synthetic data generation
- Multimodal dataset creation with question-answer pairs
- Advanced visualization and evaluation metrics
- Customizable training configurations

## Getting Started

### Installation

The recommended installation method uses conda:

```bash
# Clone the repository
git clone https://github.com/yourusername/internvl2-receipt-counter.git
cd internvl2-receipt-counter

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate internvl_env
```

For macOS users who encounter issues with specific packages:

```bash
# Install problematic packages separately
conda install -c conda-forge sentencepiece scikit-learn
```

### Model Setup

1. Download the pretrained InternVL2 model:

```bash
python utils/huggingface_model_download.py --model_name OpenGVLab/InternVL2_5-1B --output_dir ~/models/InternVL2_5-1B
```

2. Update the configuration files with your model path:

```yaml
# In config/config.yaml and config/multimodal_config.yaml
model:
  pretrained_path: "/absolute/path/to/InternVL2_5-1B"
```

## Data Generation

### Synthetic Receipt Images

Generate a dataset of synthetic receipt images:

```bash
python scripts/generate_data.py --output_dir datasets --num_collages 1000 --count_probs "0.3,0.3,0.2,0.1,0.1" --stapled_ratio 0.3 --image_size 2048
```

This creates high-resolution receipt images (2048×2048) that will be automatically resized to 448×448 during training.

### Multimodal Dataset

Create a multimodal dataset with question-answer pairs:

```bash
PYTHONPATH=. python scripts/generate_multimodal_data.py --base_dir data/raw --output_dir data/multimodal --num_samples 1000 --image_size 448
```

The multimodal dataset includes four types of question-answer pairs:
- Counting questions: "How many receipts are in this image?"
- Existence questions: "Are there any receipts visible?"
- Value questions: "What is the total value of these receipts?"
- Detail questions: "Which store has the highest receipt value?"

## Training

### Vision-Only Training

Train the model for receipt counting classification only:

```bash
python scripts/train.py --config config/config.yaml --output_dir models/vision_only
```

### Multimodal Training

Train the full vision-language multimodal model:

```bash
PYTHONPATH=. python scripts/train_multimodal.py --config config/multimodal_config.yaml --output-dir models/multimodal
```

Training implements a multi-stage approach:
1. **Stage 1**: Train with frozen vision encoder
2. **Stage 2**: Selectively unfreeze the vision encoder with low learning rate
3. **Stage 3**: End-to-end fine-tuning with balanced learning rates

## Evaluation and Testing

Evaluate the model's performance:

```bash
python scripts/evaluate.py --config config/config.yaml --model_path models/vision_only/best_model.pt
```

For multimodal evaluation:

```bash
PYTHONPATH=. python scripts/evaluate_multimodal.py --model-path models/multimodal/best_model.pt
```

Test with custom images and questions:

```bash
PYTHONPATH=. python scripts/test_multimodal_model.py --model_path models/multimodal/best_model.pt --image_path path/to/image.jpg --questions "How many receipts are in this image?" "What is the total value?"
```

## Configuration Options

The project uses YAML configuration files for flexible parameterization:

- **config/config.yaml**: Basic vision-only configuration
- **config/multimodal_config.yaml**: Full vision-language configuration
- **config/ablation_config.yaml**: Configurations for ablation studies
- **config/hyperparameter_config.yaml**: Hyperparameter optimization settings

Examples of key configuration parameters:

```yaml
# Model configuration
model:
  pretrained_path: "/path/to/model"
  use_8bit: false
  multimodal: true
  num_classes: 3

# Training configuration
training:
  epochs: 15
  learning_rate: 2.0e-5
  flash_attention: true
  fp16: false
  three_stage:
    enabled: true
    stage2:
      start_epoch: 6
    stage3:
      start_epoch: 11
```

## Project Structure

```
internvl2-receipt-counter/
├── config/                # Configuration files
├── data/                  # Dataset implementation
│   ├── data_generators/   # Synthetic data generation
│   └── dataset.py         # Dataset classes
├── models/                # Model implementation
│   ├── components/        # Model components
│   └── internvl2.py       # Main model implementation
├── training/              # Training implementation
│   ├── trainer.py         # Vision-only trainer
│   ├── multimodal_trainer.py  # Multimodal trainer
│   └── multimodal_loss.py     # Loss functions
├── evaluation/            # Evaluation metrics and visualization
├── scripts/               # Training and utility scripts
│   ├── generate_data.py           # Generate receipt data
│   ├── generate_multimodal_data.py # Generate multimodal data
│   ├── train.py                   # Vision-only training
│   ├── train_multimodal.py        # Multimodal training
│   ├── train_orchestrator.py      # Training orchestration
│   └── training_monitor.py        # Training monitoring
├── utils/                 # Utility functions
└── docs/                  # Documentation
    ├── product_requirements_document.md
    ├── vision_language_integration.md
    └── phase*_summary.md  # Phase documentation
```

## Advanced Features

### GPU Acceleration

For maximum training performance, enable these acceleration features:

1. **Flash Attention 2**:
   - Enable with `flash_attention: true` in config
   - Install with: `CUDA_HOME=/usr/local/cuda pip install flash-attn>=2.5.0`

2. **Mixed Precision Training**:
   - Enable with `fp16: true` in config
   - Best for initial training stages

3. **torch.compile**:
   - Enable with `torch_compile: true` in config
   - Requires PyTorch 2.0+

4. **Memory Optimization**:
   - 8-bit quantization: `use_8bit: true` in model config
   - Gradient accumulation: Set `gradient_accumulation_steps: N`

5. **Multi-GPU Training**:
   - DeepSpeed: `deepspeed scripts/train_multimodal.py --config config/multimodal_config.yaml`
   - DDP: `torchrun --nproc_per_node=NUM_GPUS scripts/train_multimodal.py --config config/multimodal_config.yaml`

### Hyperparameter Optimization

Run hyperparameter optimization:

```bash
python scripts/train_orchestrator.py --config config/multimodal_config.yaml --mode hyperparameter --hyperparameter-config config/hyperparameter_config.yaml
```

### Ablation Studies

Conduct ablation studies to measure component impact:

```bash
python scripts/train_orchestrator.py --config config/multimodal_config.yaml --mode ablation --ablation-config config/ablation_config.yaml
```

## Performance Metrics

Target performance metrics for the model:

| Metric | Target Value | Minimum Acceptable |
|--------|--------------|-------------------|
| Receipt Counting Accuracy | ≥95% | ≥90% |
| Query Response Accuracy | ≥90% | ≥85% |
| Processing Time (per image) | <2 seconds | <5 seconds |
| BLEU Score (for responses) | ≥0.7 | ≥0.6 |
| ROUGE Score (for responses) | ≥0.75 | ≥0.65 |

## License

MIT

## Acknowledgments

This project uses the InternVL2 model from:

"InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"

## Citation

If you use this code for your research, please cite:

```
@misc{internvl2-receipt-counter,
  author = {Your Name},
  title = {InternVL2 Receipt Counter: A Vision-Language System for Receipt Analysis},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/internvl2-receipt-counter}}
}
```