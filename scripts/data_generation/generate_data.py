#!/usr/bin/env python3
"""
Generate synthetic receipt dataset for training.

This script generates a dataset of synthetic receipt images and creates
appropriate train/val/test splits for model training.
"""
import argparse

# NOTE: This file has been updated to use the new ab initio implementation
# of receipt and tax document generation. The old implementation in
# data.data_generators has been replaced with data.data_generators_new.
import random
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to import from data/data_generators
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.generators.receipt_generator import generate_dataset as create_receipts


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate receipt dataset for training")
    parser.add_argument("--output_dir", default="datasets/synthetic_receipts", 
                      help="Output directory for generated dataset (use full path like datasets/synthetic_receipts)")
    parser.add_argument("--temp_dir", default="data/raw/temp_receipts",
                      help="Temporary directory for intermediate files (will be cleaned up)")
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
    parser.add_argument("--keep_temp", action="store_true", 
                       help="Keep temporary files after generation (default: delete)")
    return parser.parse_args()


def generate_dataset(args):
    """
    Generate the synthetic receipt dataset.
    """
    import shutil
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for intermediate files
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Set the random seed
    set_seed(args.seed)
    
    try:
        # 2. Generate receipts in temp directory first
        print("Generating synthetic receipts...")
        
        # Parse probability distribution
        count_probs = [float(p) for p in args.count_probs.split(',')]
        
        # Generate the dataset using our optimized module
        create_receipts(
            output_dir=temp_dir,  # Use temp directory for initial generation
            num_collages=args.num_collages,
            count_probs=count_probs,
            image_size=args.image_size,
            stapled_ratio=args.stapled_ratio,
            seed=args.seed
        )
        
        # 3. Copy final files from temp directory to output directory
        print(f"Moving generated files to output directory...")
        
        # Move metadata.csv file
        if (temp_dir / "metadata.csv").exists():
            shutil.copy2(temp_dir / "metadata.csv", output_dir / "metadata.csv")
        
        # Move images directory
        if (temp_dir / "images").exists():
            # Create images directory in output if it doesn't exist
            (output_dir / "images").mkdir(exist_ok=True)
            
            # Copy all image files
            for img_file in (temp_dir / "images").glob("*.png"):
                shutil.copy2(img_file, output_dir / "images" / img_file.name)
        
        # 4. Clean up temporary files if not keeping them
        if not args.keep_temp:
            print(f"Cleaning up temporary files in {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                print(f"Successfully removed temporary directory: {temp_dir}")
                
                # Also try to remove parent directory if empty
                parent_dir = temp_dir.parent
                if parent_dir.exists() and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
                    print(f"Removed empty parent directory: {parent_dir}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary files: {e}")
        
        print(f"Dataset generation complete! Images created at {args.image_size}×{args.image_size}")
        print(f"Synthetic receipts saved to {output_dir}")
        img_size = f"{args.image_size}×{args.image_size}"
        resize_note = f"Note: {img_size} images will be resized to 448×448 during training"
        print(resize_note)
        
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