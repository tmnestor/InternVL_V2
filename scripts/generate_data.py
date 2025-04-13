#!/usr/bin/env python3
"""
Generate synthetic receipt dataset for training.

This script generates a dataset of synthetic receipt images and creates
appropriate train/val/test splits for model training.
"""
import argparse
import os
import sys
from pathlib import Path

# NOTE: This file has been updated to use the new ab initio implementation
# of receipt and tax document generation. The old implementation in
# data.data_generators has been replaced with data.data_generators_new.
import random
import numpy as np

# Add parent directory to path to import from data/data_generators
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.data_generators_new.receipt_generator import generate_dataset as create_receipts


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate receipt dataset for training")
    parser.add_argument("--output_dir", default="datasets", help="Base output directory")
    parser.add_argument("--num_collages", type=int, default=300, help="Number of collages to generate")
    parser.add_argument("--count_probs", default="0.3,0.3,0.2,0.1,0.1,0", 
                      help="Probability distribution for receipt counts")
    parser.add_argument("--stapled_ratio", type=float, default=0.3,
                      help="Ratio of images that should have stapled receipts (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--image_size", type=int, default=2048, 
                      help="Output image size (default: 2048 for high-resolution receipt photos)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion for training set")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion for validation set")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion for test set")
    return parser.parse_args()


def generate_dataset(args):
    """
    Generate the synthetic receipt dataset.
    """
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Set the random seed
    set_seed(args.seed)
    
    try:
        # 2. Create the synthetic receipts directory
        print("Generating synthetic receipts...")
        synthetic_dir = output_dir / "synthetic_receipts"
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse probability distribution
        count_probs = [float(p) for p in args.count_probs.split(',')]
        
        # Generate the dataset using our optimized module
        create_receipts(
            output_dir=synthetic_dir,
            num_collages=args.num_collages,
            count_probs=count_probs,
            image_size=args.image_size,
            stapled_ratio=args.stapled_ratio,
            seed=args.seed
        )
        
        print(f"Dataset generation complete! Created images at {args.image_size}×{args.image_size} resolution")
        print(f"Synthetic receipts saved to {synthetic_dir}")
        print(f"Note: High-resolution {args.image_size}×{args.image_size} images will be resized to 448×448 during training")
        
    except Exception as e:
        print(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    
    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-10:
        print("Error: Split ratios must sum to 1.0")
        sys.exit(1)
    
    generate_dataset(args)