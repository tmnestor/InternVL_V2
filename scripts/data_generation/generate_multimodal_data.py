#!/usr/bin/env python3
"""
Generate multimodal dataset for vision-language receipt counter.

This script creates datasets for training and evaluation of the multimodal
receipt counter model with both vision and language inputs.
"""
import argparse
import random
from pathlib import Path

# NOTE: This file has been updated to use the new ab initio implementation
# of receipt and tax document generation. The old implementation in
# data.data_generators has been replaced with data.data_generators_new.
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from data.generators.create_multimodal_data import create_synthetic_multimodal_data


def prepare_datasets(
    base_data_dir,
    output_dir,
    split_sizes=(0.8, 0.1, 0.1),
    seed=42,
    image_size=448,
    num_samples=1000,
    cleanup_temp=True,
):
    """
    Create and split multimodal dataset for training, validation, and testing.
    
    Args:
        base_data_dir: Base directory for storing temporary data
        output_dir: Output directory for the split datasets
        split_sizes: Tuple of (train, val, test) split ratios
        seed: Random seed for reproducibility
        image_size: Size of the images
        num_samples: Number of samples to generate
        cleanup_temp: Whether to delete temporary data after processing
    """
    import shutil
    
    # Create directories
    base_dir = Path(base_data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} multimodal samples...")
    
    # Generate synthetic multimodal data
    multimodal_data = create_synthetic_multimodal_data(
        num_samples=num_samples,
        output_dir=base_dir / "multimodal_data",
        image_size=image_size,
        seed=seed,
    )
    
    # Group by filename to ensure same images stay in same split
    grouped = multimodal_data.groupby("filename")
    unique_filenames = list(grouped.groups.keys())
    
    # Split filenames
    train_size, val_size, test_size = split_sizes
    train_filenames, temp_filenames = train_test_split(
        unique_filenames, train_size=train_size, random_state=seed
    )
    
    # Further split temp into val and test
    val_size_adjusted = val_size / (val_size + test_size)
    val_filenames, test_filenames = train_test_split(
        temp_filenames, train_size=val_size_adjusted, random_state=seed
    )
    
    # Create output directories
    output_path = Path(output_dir)
    
    for split_name in ["train", "val", "test"]:
        (output_path / split_name / "images").mkdir(parents=True, exist_ok=True)
    
    # Get filenames for each split
    split_filenames = {
        "train": train_filenames,
        "val": val_filenames,
        "test": test_filenames,
    }
    
    # Create DataFrames for each split
    split_dfs = {}
    
    for split_name, filenames in split_filenames.items():
        # Filter rows for this split
        split_df = multimodal_data[multimodal_data["filename"].isin(filenames)]
        
        # Copy images to split directory
        for filename in filenames:
            src_path = base_dir / "multimodal_data" / "images" / filename
            dst_path = output_path / split_name / "images" / filename
            
            if src_path.exists():
                # Read and save image (this allows image processing if needed)
                img = Image.open(src_path)
                img.save(dst_path)
        
        # Save metadata CSV
        split_df.to_csv(output_path / split_name / "metadata.csv", index=False)
        split_dfs[split_name] = split_df
        
        # Print statistics
        print(f"{split_name} set: {len(filenames)} images, {len(split_df)} QA pairs")
        print(f"Receipt counts: {split_df['receipt_count'].value_counts().sort_index().to_dict()}")
    
    # Create config.json for future reference
    import json
    
    config = {
        "dataset": {
            "name": "multimodal_receipt_counter",
            "num_samples": num_samples,
            "image_size": image_size,
            "splits": {
                "train": {
                    "images": len(split_dfs["train"]["filename"].unique()),
                    "qa_pairs": len(split_dfs["train"]),
                },
                "val": {
                    "images": len(split_dfs["val"]["filename"].unique()),
                    "qa_pairs": len(split_dfs["val"]),
                },
                "test": {
                    "images": len(split_dfs["test"]["filename"].unique()),
                    "qa_pairs": len(split_dfs["test"]),
                },
            },
            "created_at": pd.Timestamp.now().isoformat(),
        }
    }
    
    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Dataset created successfully in {output_path}")
    
    # Clean up temporary data if requested
    if cleanup_temp:
        temp_data_dir = base_dir / "multimodal_data"
        if temp_data_dir.exists():
            print(f"Cleaning up temporary data in {temp_data_dir}")
            try:
                shutil.rmtree(temp_data_dir)
                print(f"Successfully removed temporary data directory: {temp_data_dir}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary data: {e}")
    
    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate multimodal receipt dataset")
    parser.add_argument("--base_dir", type=str, default="data/raw", help="Base directory for temporary raw data")
    parser.add_argument("--output_dir", type=str, default="datasets/multimodal", 
                        help="Output directory for processed datasets")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--image_size", type=int, default=448, 
                        help="Image size (default: 448 for InternVL2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--keep_temp", action="store_true", 
                        help="Keep temporary data files after processing (default: delete)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Generate dataset
    prepare_datasets(
        base_data_dir=args.base_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_samples=args.num_samples,
        seed=args.seed,
        cleanup_temp=not args.keep_temp,  # Clean up temp files unless --keep_temp is specified
    )
    
    # If we've deleted the temp data, also check if the base directory is empty and remove it
    if not args.keep_temp:
        base_dir = Path(args.base_dir)
        try:
            # Check if directory exists and is empty
            if base_dir.exists() and not any(base_dir.iterdir()):
                base_dir.rmdir()
                print(f"Removed empty base directory: {base_dir}")
        except Exception as e:
            print(f"Warning: Failed to remove base directory: {e}")