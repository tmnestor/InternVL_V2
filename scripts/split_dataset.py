#!/usr/bin/env python3
"""
Script to split the receipt dataset into train, validation, and test sets.
Ensures stratified sampling based on receipt counts to maintain class balance.
"""
import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main(args):
    """Split the dataset into train, val, and test sets with stratification."""
    print(f"Reading metadata from {args.input}")
    # Read the original metadata file
    metadata = pd.read_csv(args.input)
    
    # Convert receipt counts to classification targets (0, 1, 2+)
    metadata['class'] = metadata['receipt_count'].apply(lambda x: min(x, 2))
    
    # First split: 80% train, 20% temp (val+test)
    train_df, temp_df = train_test_split(
        metadata, 
        test_size=args.val_size + args.test_size,
        stratify=metadata['class'],
        random_state=args.seed
    )
    
    # Second split: divide temp into validation and test
    # Adjust test_size to account for the relative sizes of val and test
    val_test_ratio = args.test_size / (args.val_size + args.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_test_ratio,
        stratify=temp_df['class'],
        random_state=args.seed
    )
    
    # Clean up by removing the temporary class column
    train_df = train_df.drop(columns=['class'])
    val_df = val_df.drop(columns=['class'])
    test_df = test_df.drop(columns=['class'])
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    
    # Save the split datasets
    train_path = f"{args.output_prefix}_train.csv"
    val_path = f"{args.output_prefix}_val.csv"
    test_path = f"{args.output_prefix}_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Print statistics
    print("\nDataset split complete!")
    print(f"Train set: {len(train_df)} samples ({100 * len(train_df) / len(metadata):.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({100 * len(val_df) / len(metadata):.1f}%)")
    print(f"Test set: {len(test_df)} samples ({100 * len(test_df) / len(metadata):.1f}%)")
    
    print("\nClass distribution for training set:")
    train_class_counts = train_df['receipt_count'].apply(lambda x: min(x, 2)).value_counts().sort_index()
    for cls, count in train_class_counts.items():
        print(f"  Class {cls}: {count} samples ({100 * count / len(train_df):.1f}%)")
    
    print("\nFiles saved to:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets")
    parser.add_argument("--input", type=str, default="datasets/synthetic_receipts/metadata.csv",
                        help="Path to the input metadata CSV file")
    parser.add_argument("--output-prefix", type=str, default="datasets/synthetic_receipts/metadata",
                        help="Prefix for output CSV files (will add _train.csv, _val.csv, _test.csv)")
    parser.add_argument("--train-size", type=float, default=0.7,
                        help="Proportion of data to use for training (default: 0.7)")
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Proportion of data to use for validation (default: 0.15)")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Proportion of data to use for testing (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Verify that the proportions add up to 1
    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 1e-6:
        parser.error(f"Split proportions must add up to 1.0 (got {total})")
    
    main(args)