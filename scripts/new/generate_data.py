#!/usr/bin/env python3
"""
Generate synthetic image data for the InternVL2 Receipt Counter project.

This script generates receipt collages (with 0-5 receipts per image), including high-quality
Australian Taxation Office (ATO) documents for class 0 (zero receipts). It implements
ab initio generation rather than conversion-based approaches.
"""
# Standard library imports
import argparse
import os
import random
import sys
from pathlib import Path

# Third-party library imports
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# Setup path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Local imports (noqa comments tell linters to ignore these imports)
from data.data_generators_new.receipt_generator import create_receipt  # noqa: E402
from data.data_generators_new.tax_document_generator import create_tax_document  # noqa: E402


def create_blank_image(width, height, color="white"):
    """
    Create a blank image with specified dimensions and color.
    
    Args:
        width: Image width
        height: Image height
        color: Background color
        
    Returns:
        PIL Image object
    """
    return Image.new('RGB', (width, height), color)


def create_receipt_collage(receipt_count, image_size=2048, stapled=False):
    """
    Create a collage with a specified number of receipts.
    
    Args:
        receipt_count: Number of receipts in the image (0-5)
        image_size: Size of the output image
        stapled: Whether to create a stapled stack of receipts
        
    Returns:
        PIL Image containing receipt collage
    """
    # If no receipts, return a tax document (class 0)
    if receipt_count == 0:
        return create_tax_document(image_size)
    
    # Create blank background
    collage = create_blank_image(image_size, image_size)
    
    # Generate receipts
    receipts = []
    for _ in range(receipt_count):
        # Create a receipt with ab initio generator
        receipt = create_receipt(image_size)
        receipts.append(receipt)
    
    # For stapled receipts, create a stack with offsets
    if stapled and receipt_count > 1:
        # Find the largest receipt dimensions
        max_width = max(r.width for r in receipts)
        max_height = max(r.height for r in receipts)
        
        # Center position for the stack
        center_x = (image_size - max_width) // 2
        center_y = (image_size - max_height) // 2
        
        # Place receipts with small offsets
        for _, receipt in enumerate(receipts):
            # Calculate offset for this receipt in the stack
            offset_x = random.randint(-15, 15)
            offset_y = random.randint(-15, 15)
            
            # Ensure receipt stays within bounds
            x_pos = max(20, min(center_x + offset_x, image_size - receipt.width - 20))
            y_pos = max(20, min(center_y + offset_y, image_size - receipt.height - 20))
            
            # Paste receipt onto collage
            collage.paste(receipt, (x_pos, y_pos))
        
        # Add a staple mark
        draw = ImageDraw.Draw(collage)
        if random.random() > 0.5:  # Top staple
            staple_x = center_x + max_width // 2
            staple_y = center_y - 10
            draw.line([(staple_x-8, staple_y), (staple_x+8, staple_y)], fill="black", width=3)
            draw.line([(staple_x-5, staple_y-5), (staple_x-5, staple_y+5)], fill="black", width=3)
            draw.line([(staple_x+5, staple_y-5), (staple_x+5, staple_y+5)], fill="black", width=3)
        else:  # Side staple
            staple_x = center_x - 10
            staple_y = center_y + max_height // 2
            draw.line([(staple_x, staple_y-8), (staple_x, staple_y+8)], fill="black", width=3)
            draw.line([(staple_x-5, staple_y-5), (staple_x+5, staple_y-5)], fill="black", width=3)
            draw.line([(staple_x-5, staple_y+5), (staple_x+5, staple_y+5)], fill="black", width=3)
    else:
        # Distribute receipts across the image
        for idx, receipt in enumerate(receipts):
            if receipt_count == 1:
                # Center the receipt
                x_pos = (image_size - receipt.width) // 2
                y_pos = (image_size - receipt.height) // 2
            else:
                # Distribute receipts with some randomness
                if idx % 2 == 0:  # Left side
                    x_pos = random.randint(30, image_size // 2 - receipt.width - 30)
                else:  # Right side
                    x_pos = random.randint(image_size // 2 + 30, image_size - receipt.width - 30)
                
                y_pos = random.randint(30, image_size - receipt.height - 30)
            
            # Paste receipt onto collage
            collage.paste(receipt, (x_pos, y_pos))
            
            # Add random rotation to some receipts when there are multiple
            if receipt_count > 1 and random.random() < 0.5:
                # Create a new layer for the rotated receipt
                layer = Image.new('RGBA', (image_size, image_size), (0, 0, 0, 0))
                layer.paste(receipt, (x_pos, y_pos))
                
                # Apply rotation
                angle = random.uniform(-10, 10)
                rotated = layer.rotate(angle, resample=Image.BICUBIC, expand=False)
                
                # Composite onto main collage
                collage = Image.alpha_composite(collage.convert('RGBA'), rotated)
                collage = collage.convert('RGB')
    
    # Apply slight blur to ~20% of images for realism
    if random.random() < 0.2:
        blur_radius = random.uniform(0.3, 1.0)
        collage = collage.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Occasionally enhance contrast slightly
    if random.random() < 0.3:
        enhancer = ImageEnhance.Contrast(collage)
        collage = enhancer.enhance(random.uniform(1.0, 1.2))
    
    return collage


def split_dataset(metadata_df, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset into train, validation, and test sets while preserving class distribution.
    
    Args:
        metadata_df: Pandas DataFrame with metadata
        output_dir: Directory to save split metadata files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set (remainder goes to test)
        
    Returns:
        None (saves files to disk)
    """
    
    try:
        # Try to use sklearn for stratified splitting
        from sklearn.model_selection import train_test_split
        
        # Check if we have enough samples in each class for stratified split
        class_counts = metadata_df['receipt_count'].value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count >= 4:  # Need at least 2 samples per split (train/val/test)
            # First split into train and temp (val+test combined)
            train_df, temp_df = train_test_split(
                metadata_df, 
                train_size=train_ratio,
                stratify=metadata_df['receipt_count'],
                random_state=42
            )
            
            # Then split temp into val and test
            val_ratio_adjusted = val_ratio / (1 - train_ratio)  # Adjust for remaining portion
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_ratio_adjusted,
                stratify=temp_df['receipt_count'],
                random_state=42
            )
            print("Used sklearn for stratified dataset splitting")
        else:
            # Not enough samples for stratified split, fall back to random
            raise ValueError("Not enough samples for stratified split")
    
    except (ImportError, ValueError) as e:
        # Manual split if sklearn is not available or not enough samples
        print(f"Using non-stratified split: {e}")
        # Shuffle the dataframe
        shuffled_df = metadata_df.sample(frac=1, random_state=42)
        
        # Calculate split indices
        train_end = int(len(shuffled_df) * train_ratio)
        val_end = train_end + int(len(shuffled_df) * val_ratio)
        
        # Split the dataframe
        train_df = shuffled_df.iloc[:train_end]
        val_df = shuffled_df.iloc[train_end:val_end]
        test_df = shuffled_df.iloc[val_end:]
    
    # Save split metadata
    train_df.to_csv(output_dir / "metadata_train.csv", index=False)
    val_df.to_csv(output_dir / "metadata_val.csv", index=False)
    test_df.to_csv(output_dir / "metadata_test.csv", index=False)
    
    # Print split statistics
    print("\nDataset split statistics:")
    print(f"  Training set:   {len(train_df)} images ({train_ratio:.1%})")
    print(f"  Validation set: {len(val_df)} images ({val_ratio:.1%})")
    print(f"  Test set:       {len(test_df)} images ({1-train_ratio-val_ratio:.1%})")
    
    # Print class distribution in each split
    print("\nReceipt count distribution:")
    train_dist = train_df['receipt_count'].value_counts().sort_index()
    val_dist = val_df['receipt_count'].value_counts().sort_index()
    test_dist = test_df['receipt_count'].value_counts().sort_index()
    
    for count in sorted(metadata_df['receipt_count'].unique()):
        train_count = train_dist.get(count, 0)
        val_count = val_dist.get(count, 0)
        test_count = test_dist.get(count, 0)
        total_count = train_count + val_count + test_count
        
        print(f"  Class {count}: {train_count} train, {val_count} val, {test_count} test "
              f"(total: {total_count})")


def generate_dataset(output_dir, num_collages=300, count_probs=None, image_size=2048, 
                    stapled_ratio=0.0, seed=42):
    """
    Generate a dataset of receipt collages with varying receipt counts.
    
    Args:
        output_dir: Directory to save generated images and metadata
        num_collages: Number of collage images to generate
        count_probs: Probability distribution for receipt counts (0-5)
        image_size: Size of output images
        stapled_ratio: Ratio of images with stapled receipts (only applies to receipt_count > 1)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with image filenames and receipt counts
    """
    import pandas as pd
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Default distribution if not provided
    if count_probs is None:
        count_probs = [0.3, 0.3, 0.2, 0.1, 0.1, 0.0]  # Probabilities for 0, 1, 2, 3, 4, 5 receipts
    
    # Normalize probabilities
    count_probs = np.array(count_probs)
    count_probs = count_probs / count_probs.sum()
    
    # Generate collages
    data = []
    
    for i in range(num_collages):
        # Determine number of receipts based on probability distribution
        receipt_count = np.random.choice(len(count_probs), p=count_probs)
        
        # Determine if this should be stapled (only for multiple receipts)
        is_stapled = False
        if receipt_count > 1 and random.random() < stapled_ratio:
            is_stapled = True
        
        # Create collage
        try:
            collage = create_receipt_collage(receipt_count, image_size, stapled=is_stapled)
            
            # Save image
            filename = f"receipt_collage_{i:05d}.png"
            collage.save(images_dir / filename)
            
            # Add to dataset
            data.append({
                "filename": filename,
                "receipt_count": receipt_count,
                "is_stapled": is_stapled
            })
            
            # Progress update
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Generated {i + 1}/{num_collages} collages")
                
        except Exception as e:
            print(f"Error generating collage {i}: {e}")
    
    # Create and save metadata
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    # Print statistics
    print(f"\nDataset generation complete: {num_collages} images")
    print(f"Distribution of receipt counts: {df['receipt_count'].value_counts().sort_index()}")
    print(f"ATO tax documents (0 receipts): {len(df[df['receipt_count'] == 0])}")
    print(f"High-resolution images: {image_size}×{image_size}")
    
    if stapled_ratio > 0:
        stapled_count = df['is_stapled'].sum()
        print(f"Stapled receipts: {stapled_count} ({stapled_count/num_collages:.1%})")
    
    # Create train/val/test splits
    split_dataset(df, output_dir)
    
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic receipt dataset")
    parser.add_argument("--output_dir", default="datasets/synthetic_receipts", 
                      help="Output directory")
    parser.add_argument("--num_collages", type=int, default=300, 
                      help="Number of collages to generate")
    parser.add_argument("--count_probs", default="0.3,0.3,0.2,0.1,0.1,0", 
                      help="Probability distribution for receipt counts (0,1,2,3,4,5)")
    parser.add_argument("--image_size", type=int, default=2048,
                      help="Size of output images")
    parser.add_argument("--stapled_ratio", type=float, default=0.3, 
                      help="Ratio of images with stapled receipts")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Parse probability distribution
    count_probs = [float(p) for p in args.count_probs.split(',')]
    
    # Generate dataset
    df = generate_dataset(
        output_dir=args.output_dir,
        num_collages=args.num_collages,
        count_probs=count_probs,
        image_size=args.image_size,
        stapled_ratio=args.stapled_ratio,
        seed=args.seed
    )