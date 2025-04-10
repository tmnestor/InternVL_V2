#!/usr/bin/env python3
"""
Split receipt collage dataset into training and validation sets.

This script takes a directory of receipt collage images and their metadata
and creates train/val/test splits for model training.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def process_image(image_path, output_path, image_size=448):
    """
    Process and resize an image to the target size.
    
    Args:
        image_path: Path to source image
        output_path: Path to save processed image
        image_size: Target image size (default: 448 for InternVL2)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    img = img.resize((image_size, image_size), Image.LANCZOS)
    
    # Save processed image
    img.save(output_path)


def create_dataset_split(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, 
                       test_ratio=0.15, seed=42, image_size=448):
    """
    Create train/val/test splits from a collage dataset.
    
    Args:
        source_dir: Directory containing collage images and metadata
        output_dir: Directory to save the split dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
        image_size: Target image size for processed images
        
    Returns:
        Dictionary with train/val/test DataFrames
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Convert paths to Path objects
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = source_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
    df = pd.read_csv(metadata_path)
    
    # Perform train/val/test split
    train_val_files, test_files = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df['receipt_count']
    )
    
    train_files, val_files = train_test_split(
        train_val_files, 
        test_size=val_ratio/(train_ratio+val_ratio), 
        random_state=seed, 
        stratify=train_val_files['receipt_count']
    )
    
    # Create new DataFrames for each split
    split_dfs = {
        'train': train_files.reset_index(drop=True),
        'val': val_files.reset_index(drop=True),
        'test': test_files.reset_index(drop=True)
    }
    
    # Process and save images for each split
    images_dir = source_dir / "images"
    
    for split, split_df in split_dfs.items():
        split_dir = output_dir / split
        
        for idx, row in split_df.iterrows():
            # Source and target image paths
            source_path = images_dir / row['filename']
            target_path = split_dir / row['filename']
            
            # Process and save image
            process_image(source_path, target_path, image_size)
            
        # Save metadata for this split
        split_df.to_csv(output_dir / f"{split}.csv", index=False)
        
        print(f"{split} split: {len(split_df)} images")
        print(f"Receipt count distribution: {split_df['receipt_count'].value_counts().sort_index()}")
    
    return split_dfs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create dataset splits from receipt collages")
    parser.add_argument("--collage_dir", required=True, help="Directory with receipt collages and metadata")
    parser.add_argument("--output_dir", default="receipt_dataset", help="Output directory for dataset splits")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Proportion of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Proportion of data for validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Proportion of data for testing")
    parser.add_argument("--image_size", type=int, default=448,
                       help="Size of output images (default: 448 for InternVL2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-10:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Create dataset splits
    split_dfs = create_dataset_split(
        source_dir=args.collage_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        image_size=args.image_size
    )
    
    print("Dataset splits created successfully")