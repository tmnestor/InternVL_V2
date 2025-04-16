#!/bin/bash
# Run the basic image check script

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Activate conda environment and run the script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internvl_env
python scripts/basic_image_check.py

echo "Basic image check complete. Results saved to datasets/basic_analysis_results/"