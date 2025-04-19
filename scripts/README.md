# Scripts

This directory contains executable scripts for training, evaluation, and data generation.

## Directory Structure

- `classification/`: Scripts for question classification tasks
  - `train_enhanced_classifier.py`: Trains the question classifier model
  - `test_enhanced_classifier.py`: Evaluates the question classifier model
- `data_generation/`: Scripts for synthetic data generation
  - `generate_data.py`: Generates synthetic receipt and tax document data
  - `generate_multimodal_data.py`: Creates multimodal dataset with images and text
  - `generate_unified_data.py`: Creates unified datasets for both receipt counting and QA tasks
  - `legacy/`: Deprecated scripts that have been replaced by newer implementations
    - `standalone_generator.py`: (Legacy) Simplified generator with minimal dependencies
    - `visual_compare.py`: (Legacy) Compares original vs ab initio implementations
- `training/`: Scripts for model training and evaluation
  - `train_unified_multimodal.py`: Trains the unified InternVL multimodal model (recommended)
  - `evaluate_multimodal.py`: Evaluates the multimodal model performance
  - `legacy/`: Deprecated scripts that have been replaced by newer implementations
    - `train_multimodal.py`: (Legacy) Original multimodal training script
    - `train_orchestrator.py`: (Legacy) Multi-stage training orchestration
    - `training_monitor.py`: (Legacy) Training monitoring dashboard
- `evaluate.py`: General evaluation script that works with various models
- `split_dataset.py`: Splits datasets into train/val/test sets
- `check_gpu.py`: Checks GPU availability and configuration
- `test_multimodal_model.py`: Tests the multimodal model on input images
- `utils/huggingface_model_download.py`: Downloads models from Huggingface Hub

## Usage

To run a script, use Python with PYTHONPATH set to the project root:

```bash
# Classification
PYTHONPATH=. python scripts/classification/train_enhanced_classifier.py --num-epochs 15
PYTHONPATH=. python scripts/classification/test_enhanced_classifier.py --model-path models/enhanced_classifier/best_model.pt

# Data Generation
PYTHONPATH=. python scripts/data_generation/generate_data.py --output_dir datasets/synthetic_receipts
PYTHONPATH=. python scripts/data_generation/generate_unified_data.py --output_dir data/unified_dataset
# Legacy script (deprecated):
# PYTHONPATH=. python scripts/data_generation/legacy/standalone_generator.py --num_samples 10

# Model Training
PYTHONPATH=. python scripts/training/train_unified_multimodal.py --config config/model/unified_multimodal_config.yaml --output-dir models/unified_multimodal
PYTHONPATH=. python scripts/training/evaluate_multimodal.py --model-path models/unified_multimodal/best_model.pt

# Legacy training (not recommended)
# PYTHONPATH=. python scripts/training/legacy/train_multimodal.py --config legacy/config/model/multimodal_config.yaml

# Utilities
PYTHONPATH=. python scripts/split_dataset.py --input data/raw --output data/processed
PYTHONPATH=. python scripts/check_gpu.py
```

## Note on Output Directories

All script outputs (logs, model checkpoints, evaluation results) are now stored in the `archive` directory:
- Logs are saved to `archive/logging/logs/`
- Model outputs and checkpoints are saved to `models/` directory
- Evaluation results are saved to `archive/output/`
