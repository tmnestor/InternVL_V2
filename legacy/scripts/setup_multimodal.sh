#!/bin/bash
# Setup script for multimodal improvement implementation

# Create necessary directories
mkdir -p data/question_data
mkdir -p models/question_classifier
mkdir -p logs

# Check if conda environment exists
if conda env list | grep -q "internvl_env"; then
    echo "Using existing conda environment: internvl_env"
else
    echo "Creating conda environment from environment.yml"
    conda env create -f environment.yml
fi

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internvl_env

# Install additional dependencies
pip install transformers==4.31.0 datasets==2.14.5 evaluate==0.4.0 accelerate==0.22.0

# Prepare question dataset directories
python -c "
import os
import json
from pathlib import Path

# Default question datasets
question_types = ['DOCUMENT_TYPE', 'COUNTING', 'DETAIL_EXTRACTION', 'PAYMENT_INFO', 'TAX_INFO']

for split in ['train', 'val', 'test']:
    dataset_path = Path('data/question_data') / f'question_dataset_{split}.json'
    if not dataset_path.exists():
        print(f'Creating default {split} dataset')
        examples = []
        for qtype in question_types:
            for i in range(5):  # 5 examples per type per split
                examples.append({
                    'question': f'Sample {qtype.lower()} question {i+1}. Replace with real questions.',
                    'type': qtype
                })
        os.makedirs(dataset_path.parent, exist_ok=True)
        with open(dataset_path, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f'Created {dataset_path} with {len(examples)} example questions')
"

echo ""
echo "Setup completed successfully!"
echo ""
echo "To train the question classifier:"
echo "PYTHONPATH=. conda run -n internvl_env python scripts/train_question_classifier.py"
echo ""
echo "To train the multimodal model:"
echo "PYTHONPATH=. conda run -n internvl_env python scripts/train_multimodal.py"
echo ""
echo "For Linux development environment:"
echo "source setup.sh && python scripts/train_question_classifier.py"
echo "source setup.sh && python scripts/train_multimodal.py"