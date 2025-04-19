#!/bin/bash
# Setup script for unified multitask implementation with InternVL2

# Print header
echo "======================================================"
echo "🚀 Setting up environment for InternVL2 Unified Model"
echo "======================================================"

# Create necessary directories
mkdir -p data/unified_dataset
mkdir -p models/unified_multimodal
mkdir -p logs
mkdir -p results/unified_multimodal/tensorboard

# Check if conda environment exists
if conda env list | grep -q "internvl_env"; then
    echo "✅ Using existing conda environment: internvl_env"
else
    echo "🔄 Creating conda environment from environment.yml"
    conda env create -f environment.yml
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate internvl_env

# Install/update dependencies
echo "🔄 Installing required dependencies..."
pip install transformers==4.31.0 datasets==2.14.5 evaluate==0.4.0 accelerate==0.22.0 torch==2.0.1

# Set environment variables
export TOKENIZERS_PARALLELISM=false

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Quick Start Guide:"
echo ""
echo "1️⃣ Generate unified dataset:"
echo "   PYTHONPATH=. conda run -n internvl_env python scripts/data_generation/generate_unified_data.py --output_dir data/unified_dataset --num_samples 1000"
echo ""
echo "2️⃣ Train the unified multitask model:"
echo "   PYTHONPATH=. conda run -n internvl_env python scripts/training/train_unified_multimodal.py --config config/model/unified_multimodal_config.yaml"
echo ""
echo "3️⃣ Evaluate the model:"
echo "   PYTHONPATH=. conda run -n internvl_env python scripts/training/evaluate_multimodal.py --model-path models/unified_multimodal/best_model.pt"
echo ""
echo "For Linux development environment:"
echo "   source setup.sh && python scripts/data_generation/generate_unified_data.py --output_dir data/unified_dataset --num_samples 1000"
echo "   source setup.sh && python scripts/training/train_unified_multimodal.py --config config/model/unified_multimodal_config.yaml"
echo "======================================================"