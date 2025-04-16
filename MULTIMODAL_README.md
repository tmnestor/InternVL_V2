# InternVL_V2 Multimodal Improvement Implementation

This README provides instructions for implementing the multimodal improvement strategy as outlined in the `multimodal_improvement_strategy.md` document.

## Overview

The implementation introduces several key enhancements to the multimodal system:

1. **Question Classifier**: Analyzes and categorizes questions about receipts and tax documents
2. **Enhanced Template System**: Provides more diverse and context-specific responses
3. **Detail Extractor**: Identifies specific details in documents for improved response generation
4. **Improved Loss Function**: Implements weighted, dynamic, and contrastive losses for better training

## Implementation Components

The implementation consists of these key components:

- `models/components/question_classifier.py`: Classifies questions into different categories
- `models/components/template_system.py`: Structured, hierarchical template system for responses
- `models/components/detail_extractor.py`: Extracts specific details from document images
- `training/multimodal_loss.py`: Enhanced loss function with multiple components
- `data/question_dataset.py`: Dataset for training the question classifier
- `scripts/train_question_classifier.py`: Script for training the question classifier

## Setup Instructions

### Prerequisites

- Conda environment set up as specified in `environment.yml`
- Access to a GPU-enabled machine for training

### Installation

1. Run the setup script to prepare the environment and create necessary directories:

```bash
# On macOS
bash setup_multimodal.sh

# On Linux
chmod +x setup_multimodal.sh
./setup_multimodal.sh
```

2. Prepare question datasets by either:
   - Using the default datasets created by the setup script
   - Creating custom datasets following the format in `data/question_dataset.py`

## Training Pipeline

### 1. Train the Question Classifier

```bash
# macOS with Homebrew:
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/internvl_env/bin/python scripts/train_question_classifier.py

# Alternative using conda run (if working):
PYTHONPATH=. conda run -n internvl_env python scripts/train_question_classifier.py
  
# Linux:
source setup.sh && python scripts/train_question_classifier.py
```

### 2. Train the Full Multimodal Model

```bash
# macOS with Homebrew:
PYTHONPATH=. /opt/homebrew/Caskroom/miniforge/base/envs/internvl_env/bin/python scripts/train_multimodal.py --config config/multimodal_config.yaml --output-dir models/multimodal

# Alternative using conda run (if working):
PYTHONPATH=. conda run -n internvl_env python scripts/train_multimodal.py --config config/multimodal_config.yaml --output-dir models/multimodal
  
# Linux:
source setup.sh && python scripts/train_multimodal.py --config config/multimodal_config.yaml --output-dir models/multimodal
```

## Two-Stage Training Strategy

The system implements the two-stage training strategy specified in the improvement document:

1. **Stage 1**: Train vision encoder and classification head
   - Initialize the model with pretrained weights
   - Freeze language model components
   - Train only on document classification task

2. **Stage 2**: Train text generation with teacher forcing
   - Freeze vision components or use a very small learning rate
   - Focus on improving language generation
   - Utilize the enhanced loss function with all components

## Evaluation

The improved system can be evaluated using:

1. **Classification Accuracy**: Document type and receipt count accuracy
2. **Response Quality**:
   - BLEU score
   - Token overlap
   - Exact match percentage
   - ROUGE-L score
3. **Question Understanding**:
   - Question classification accuracy
   - Answer relevance rating
4. **Detail Extraction**:
   - Named entity recognition accuracy
   - Value extraction accuracy

## Known Issues

- Set `TOKENIZERS_PARALLELISM=false` when using HuggingFace tokenizers with multiprocessing
- Reduce `num_workers` to 1 in config files when encountering tokenizer parallelism warnings
- Disable `flash_attention` in the config file if not properly installed