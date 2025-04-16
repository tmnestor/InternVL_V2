#!/bin/bash
# Run the tax document verification script

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Activate conda environment and run the script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internvl_env
python scripts/verify_tax_documents.py

echo "Verification complete. Results saved to datasets/verification_results/"