#!/usr/bin/env python3
"""
Generate unified dataset for receipt counting and vision-language tasks.

This script combines receipt counting and question-answering capabilities
into a single unified dataset structure.
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from data.generators.create_multimodal_data import create_synthetic_multimodal_data


def create_unified_dataset(
    output_dir,
    receipt_dataset_dir=None,
    num_samples=1000,
    image_size=448,
    split_sizes=(0.8, 0.1, 0.1),
    seed=42,
    cleanup_temp=True,
):
    """
    Create a unified dataset with both receipt counting and QA data.
    
    Args:
        output_dir: Output directory for the unified dataset
        receipt_dataset_dir: Optional directory containing existing receipt dataset
        num_samples: Number of samples to generate if not using existing data
        image_size: Image size for newly generated samples
        split_sizes: Tuple of (train, val, test) split ratios
        seed: Random seed for reproducibility
        cleanup_temp: Whether to delete temporary files
    """
    print(f"Creating unified dataset in {output_dir}")
    output_path = Path(output_dir)
    
    # Generate synthetic data if no existing receipt dataset is provided
    if receipt_dataset_dir is None:
        print(f"Generating {num_samples} new multimodal samples...")
        
        # Create temporary directory for multimodal data generation
        temp_dir = Path("temp_data")
        temp_dir.mkdir(exist_ok=True)
        
        # Generate multimodal data with questions and answers
        multimodal_data = create_synthetic_multimodal_data(
            num_samples=num_samples,
            output_dir=temp_dir / "multimodal_data",
            image_size=image_size,
            seed=seed,
        )
    else:
        print(f"Using existing receipt dataset from {receipt_dataset_dir}")
        receipt_dataset_path = Path(receipt_dataset_dir)
        
        # Check if this is a multimodal dataset with questions
        metadata_path = receipt_dataset_path / "train" / "metadata.csv"
        if metadata_path.exists():
            # This is already in the right format with train/val/test
            if "question" in pd.read_csv(metadata_path).columns:
                print("This is already a multimodal dataset with questions. Copying directly...")
                
                # Create the unified dataset by copying the existing dataset
                for split in ["train", "val", "test"]:
                    split_dir = receipt_dataset_path / split
                    if split_dir.exists():
                        # Create output directory
                        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
                        
                        # Copy metadata
                        shutil.copy(
                            split_dir / "metadata.csv",
                            output_path / split / "metadata.csv"
                        )
                        
                        # Copy images
                        images_dir = split_dir / "images"
                        for image_file in images_dir.glob("*.png"):
                            shutil.copy(
                                image_file,
                                output_path / split / "images" / image_file.name
                            )
                
                print("Dataset copied successfully")
                return
            else:
                # This is not a multimodal dataset, but it might already be split
                if (receipt_dataset_path / "metadata_train.csv").exists():
                    # Load existing splits
                    train_df = pd.read_csv(receipt_dataset_path / "metadata_train.csv")
                    val_df = pd.read_csv(receipt_dataset_path / "metadata_val.csv")
                    test_df = pd.read_csv(receipt_dataset_path / "metadata_test.csv")
                    
                    # Check if it has receipt_count column
                    if "receipt_count" in train_df.columns:
                        print("Using existing split from receipt dataset")
                        temp_dir = receipt_dataset_path
                        
                        # We'll just add question-answer pairs to these
                        # Generate multimodal data similar to receipt images
                        multimodal_data = create_synthetic_multimodal_data(
                            num_samples=len(train_df) + len(val_df) + len(test_df),
                            output_dir=temp_dir / "temp_multimodal",
                            image_size=image_size,
                            seed=seed,
                        )
                    else:
                        # This dataset doesn't have receipt counts
                        print("Error: Incompatible dataset format. Missing receipt_count column.")
                        return
                elif (receipt_dataset_path / "metadata.csv").exists():
                    # Only has a single metadata file, we need to split it
                    print("Using unsplit receipt dataset and creating splits")
                    df = pd.read_csv(receipt_dataset_path / "metadata.csv")
                    
                    # Group by filename
                    grouped = df.groupby("filename")
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
                    
                    # Create dataframes for each split
                    train_df = df[df["filename"].isin(train_filenames)]
                    val_df = df[df["filename"].isin(val_filenames)]
                    test_df = df[df["filename"].isin(test_filenames)]
                    
                    # Setup temporary directory
                    temp_dir = receipt_dataset_path
                    
                    # Generate multimodal data similar to receipt images
                    multimodal_data = create_synthetic_multimodal_data(
                        num_samples=len(df),
                        output_dir=temp_dir / "temp_multimodal",
                        image_size=image_size,
                        seed=seed,
                    )
                else:
                    print("Error: Invalid receipt dataset directory structure")
                    return
        else:
            print("Error: Invalid receipt dataset directory structure")
            return
    
    # Create output directories
    for split_name in ["train", "val", "test"]:
        (output_path / split_name / "images").mkdir(parents=True, exist_ok=True)
    
    # Group by filename to ensure the same images stay in the same split
    if "temp_multimodal" in str(temp_dir):
        # Using existing receipt dataset with added QA pairs
        grouped = multimodal_data.groupby("filename")
        qa_data = {}
        
        # Extract questions and answers for each filename
        for filename, group in grouped:
            qa_data[filename] = group[["question", "answer", "qa_pair_idx"]].to_dict(orient="records")
        
        # Process each split
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            split_output = []
            
            # Add questions to each receipt entry
            for _, row in split_df.iterrows():
                filename = row["filename"]
                receipt_count = row["receipt_count"]
                
                # Use the same filename to get QA pairs
                # If filename doesn't exist in multimodal data, generate random QA pairs
                if filename in qa_data:
                    qa_pairs = qa_data[filename]
                else:
                    # Generate random QA pairs
                    # Find a random file with the same receipt count
                    same_count_files = [f for f, g in grouped.groups.items() 
                                       if multimodal_data.loc[g[0], "receipt_count"] == receipt_count]
                    if same_count_files:
                        # Use QA pairs from a similar file
                        random_file = random.choice(same_count_files)
                        qa_pairs = qa_data[random_file]
                    else:
                        # Use QA pairs from any file
                        random_file = random.choice(list(qa_data.keys()))
                        qa_pairs = qa_data[random_file]
                
                # Add QA pairs to the output
                for qa_pair in qa_pairs:
                    split_output.append({
                        "filename": filename,
                        "receipt_count": receipt_count,
                        "question": qa_pair["question"],
                        "answer": qa_pair["answer"],
                        "qa_pair_idx": qa_pair["qa_pair_idx"]
                    })
            
            # Create a dataframe for this split
            split_output_df = pd.DataFrame(split_output)
            
            # Save metadata CSV
            split_output_df.to_csv(output_path / split_name / "metadata.csv", index=False)
            
            # Copy images
            for filename in split_df["filename"].unique():
                src_path = temp_dir / "images" / filename
                dst_path = output_path / split_name / "images" / filename
                
                if src_path.exists():
                    # Read and save image (this allows image processing if needed)
                    img = Image.open(src_path)
                    img.save(dst_path)
            
            print(f"{split_name} set: {len(split_df['filename'].unique())} images, {len(split_output_df)} QA pairs")
            print(f"Receipt counts: {split_output_df['receipt_count'].value_counts().sort_index().to_dict()}")
    else:
        # Creating entirely new dataset
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
                src_path = temp_dir / "multimodal_data" / "images" / filename
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
    
    # Create dataset info file
    import json
    
    config = {
        "dataset": {
            "name": "unified_receipt_counter",
            "num_samples": num_samples,
            "image_size": image_size,
            "splits": {
                "train": {
                    "images": len(split_filenames["train"]) if "split_filenames" in locals() else len(train_df["filename"].unique()),
                    "qa_pairs": len(split_dfs["train"]) if "split_dfs" in locals() else len(split_output_df),
                },
                "val": {
                    "images": len(split_filenames["val"]) if "split_filenames" in locals() else len(val_df["filename"].unique()),
                    "qa_pairs": len(split_dfs["val"]) if "split_dfs" in locals() else len(split_output_df),
                },
                "test": {
                    "images": len(split_filenames["test"]) if "split_filenames" in locals() else len(test_df["filename"].unique()),
                    "qa_pairs": len(split_dfs["test"]) if "split_dfs" in locals() else len(split_output_df),
                },
            },
            "created_at": pd.Timestamp.now().isoformat(),
        }
    }
    
    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Unified dataset created successfully in {output_path}")
    
    # Clean up temporary data if requested
    if cleanup_temp and "temp_multimodal" in str(temp_dir):
        temp_data_dir = temp_dir / "temp_multimodal"
        if temp_data_dir.exists():
            print(f"Cleaning up temporary data in {temp_data_dir}")
            try:
                shutil.rmtree(temp_data_dir)
                print(f"Successfully removed temporary data directory: {temp_data_dir}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary data: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate unified receipt dataset")
    parser.add_argument("--receipt_data", type=str, default=None, 
                        help="Optional existing receipt dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/unified_dataset", 
                        help="Output directory for unified dataset")
    parser.add_argument("--num_samples", type=int, default=1000, 
                        help="Number of samples to generate (if not using existing data)")
    parser.add_argument("--image_size", type=int, default=448, 
                        help="Image size (default: 448 for InternVL2)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--keep_temp", action="store_true", 
                        help="Keep temporary data files (default: delete)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create the unified dataset
    create_unified_dataset(
        output_dir=args.output_dir,
        receipt_dataset_dir=args.receipt_data,
        num_samples=args.num_samples,
        image_size=args.image_size,
        seed=args.seed,
        cleanup_temp=not args.keep_temp,
    )