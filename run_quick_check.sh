#!/bin/bash
# Run the quick visual check script

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Activate conda environment and run the script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate internvl_env
python scripts/quick_visual_check.py

echo "Quick visual check complete. Results saved to datasets/quick_analysis_results/"