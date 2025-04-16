#!/bin/bash
# Wrapper script to run split_dataset.py with the conda environment

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Activate conda environment and run the script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internvl_env
python scripts/split_dataset.py

# Print config update instructions
echo ""
echo "Now update config/config.yaml with the following paths:"
echo "  train_csv: datasets/synthetic_receipts/metadata_train.csv"
echo "  val_csv: datasets/synthetic_receipts/metadata_val.csv"
echo "  test_csv: datasets/synthetic_receipts/metadata_test.csv"