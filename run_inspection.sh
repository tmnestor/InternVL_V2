#!/bin/bash
# Run the class 0 image inspection script

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Activate conda environment and run the script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internvl_env
python scripts/inspect_class0_images.py

echo "Inspection complete. Results saved to datasets/inspection_results/"