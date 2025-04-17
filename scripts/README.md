# Scripts

This directory contains executable scripts for training, evaluation, and data generation.

## Directory Structure

- `classification/`: Scripts for question classification tasks
  - `train_enhanced_classifier.py`: Trains the question classifier model
  - `test_enhanced_classifier.py`: Evaluates the question classifier model
- `data_generation/`: Scripts for synthetic data generation
  - `generate_data.py`: Generates synthetic receipt and tax document data
  - `generate_multimodal_data.py`: Creates multimodal dataset with images and text
  - `standalone_generator.py`: Simplified generator with minimal dependencies
  - `visual_compare.py`: Compares original vs ab initio implementations
- `training/`: Scripts for model training and evaluation
  - `train_multimodal.py`: Trains the InternVL multimodal model
  - `evaluate_multimodal.py`: Evaluates the multimodal model performance
  - `train_orchestrator.py`: Orchestrates the multi-stage training process
  - `training_monitor.py`: Monitors training metrics and progress
- `evaluate.py`: General evaluation script that works with various models
- `split_dataset.py`: Splits datasets into train/val/test sets
- `check_gpu.py`: Checks GPU availability and configuration
- `huggingface_model_download.py`: Downloads models from Huggingface Hub
- `test_multimodal_model.py`: Tests the multimodal model on input images

## Usage

To run a script, use Python with PYTHONPATH set to the project root:

```bash
# Classification
PYTHONPATH=. python scripts/classification/train_enhanced_classifier.py --num-epochs 15
PYTHONPATH=. python scripts/classification/test_enhanced_classifier.py --model-path models/enhanced_classifier/best_model.pt

# Data Generation
PYTHONPATH=. python scripts/data_generation/generate_data.py --output_dir datasets/synthetic_receipts
PYTHONPATH=. python scripts/data_generation/standalone_generator.py --num_samples 10

# Model Training
PYTHONPATH=. python scripts/training/train_multimodal.py --config config/model/multimodal_config.yaml
PYTHONPATH=. python scripts/training/evaluate_multimodal.py --model-path models/multimodal/best_model.pt

# Utilities
PYTHONPATH=. python scripts/split_dataset.py --input data/raw --output data/processed
PYTHONPATH=. python scripts/check_gpu.py
```

## Note on Output Directories

All script outputs (logs, model checkpoints, evaluation results) are now stored in the `archive` directory:
- Logs are saved to `archive/logging/logs/`
- Model outputs and checkpoints are saved to `models/` directory
- Evaluation results are saved to `archive/output/`
