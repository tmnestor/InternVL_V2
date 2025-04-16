#!/bin/bash

# setup.sh - User-specific setup script
# Usage: source setup.sh [working_directory] [conda_env_name]

# Default values
DEFAULT_DIR="$HOME/nfs_share/tod/internvl-receipt-counter"
DEFAULT_ENV="internvl_env"

# Parse arguments
WORK_DIR=${1:-$DEFAULT_DIR}
CONDA_ENV=${2:-$DEFAULT_ENV}

# Print header
echo "========================================"
echo "🚀 Setting up environment for internvl-receipt-counter"
echo "========================================"

# Change to working directory
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
    echo "✅ Changed directory to: $(pwd)"
else
    echo "❌ Error: Directory $WORK_DIR does not exist"
    return 1  # Use return instead of exit when sourced
fi

# Activate conda
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
    echo "✅ Conda initialized"
    
    # Try to activate the conda environment
    if conda activate "$CONDA_ENV" 2>/dev/null; then
        echo "✅ Activated conda environment: $CONDA_ENV"
    else
        echo "⚠️ Could not activate conda environment: $CONDA_ENV"
        echo "   Available environments:"
        conda env list
        return 1
    fi
else
    echo "❌ Error: Conda initialization file not found"
    return 1
fi

# Set up aliases for common commands
alias train="python main.py --config config/config.yaml --mode train"
alias evaluate="python main.py --config config/config.yaml --mode evaluate"

echo "✅ Set up shortcuts:"
echo "   - train: Run training"
echo "   - evaluate: Run evaluation"
echo ""
echo "🔍 Current status:"
echo "   - Working directory: $(pwd)"
echo "   - Python: $(which python)"
echo "   - Conda environment: $CONDA_ENV"
echo "========================================"
echo "To use this environment, remember to run with 'source':"
echo "source setup.sh [directory] [environment]"
echo "========================================"