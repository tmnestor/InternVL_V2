#!/usr/bin/env python3
"""
Create receipt collages from existing receipt images.

This script creates receipt collages by combining existing receipt images
in various arrangements to serve as training data for the receipt counter model.
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from data.data_generators.receipt_processor import create_blank_image


def load_receipt_images(source_dir):
    """
    Load receipt images from a source directory.
    
    Args:
        source_dir: Directory containing individual receipt images
        
    Returns:
        List of PIL Images
    """
    source_dir = Path(source_dir)
    image_paths = list(source_dir.glob("*.png")) + list(source_dir.glob("*.jpg"))
    
    receipts = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            receipts.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    print(f"Loaded {len(receipts)} receipt images from {source_dir}")
    return receipts


def create_collage(receipt_images, receipt_count, image_size=448):
    """
    Create a collage with a specified number of receipts.
    
    Args:
        receipt_images: List of receipt images to sample from
        receipt_count: Number of receipts to include
        image_size: Size of the output collage image
        
    Returns:
        PIL Image containing the receipt collage
    """
    # Create blank image for collage
    collage = create_blank_image(image_size, image_size, 'white')
    
    # If no receipts, return blank image with some noise
    if receipt_count == 0:
        # Add some random noise/texture to avoid completely blank images
        draw = ImageDraw.Draw(collage)
        for _ in range(100):
            x1 = random.randint(0, image_size - 1)
            y1 = random.randint(0, image_size - 1)
            x2 = min(x1 + random.randint(1, 10), image_size - 1)
            y2 = min(y1 + random.randint(1, 10), image_size - 1)
            color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        return collage
    
    # Randomly sample receipts
    if receipt_count > len(receipt_images):
        # If we need more receipts than available, sample with replacement
        selected_receipts = random.choices(receipt_images, k=receipt_count)
    else:
        # Sample without replacement
        selected_receipts = random.sample(receipt_images, receipt_count)
    
    # Resize receipts to fit in collage
    processed_receipts = []
    for receipt in selected_receipts:
        # Calculate resize factor based on receipt count
        max_width = image_size // min(receipt_count, 2)
        max_height = image_size // min(receipt_count, 2)
        
        # Resize receipt, maintaining aspect ratio
        scale_factor = min(max_width / receipt.width, max_height / receipt.height)
        new_width = int(receipt.width * scale_factor)
        new_height = int(receipt.height * scale_factor)
        
        # Ensure minimum size
        new_width = max(new_width, 50)
        new_height = max(new_height, 100)
        
        # Resize and add random rotation
        resized = receipt.resize((new_width, new_height), Image.LANCZOS)
        if random.random() > 0.5:
            rotation = random.uniform(-10, 10)
            resized = resized.rotate(rotation, expand=True, fillcolor='white')
        
        processed_receipts.append(resized)
    
    # Place receipts in collage
    for idx, receipt in enumerate(processed_receipts):
        if receipt_count == 1:
            # Center the receipt
            x_pos = (image_size - receipt.width) // 2
            y_pos = (image_size - receipt.height) // 2
        else:
            # Distribute receipts randomly, avoiding edges
            x_pos = random.randint(10, image_size - receipt.width - 10)
            y_pos = random.randint(10, image_size - receipt.height - 10)
        
        # Paste the receipt onto the collage
        collage.paste(receipt, (x_pos, y_pos))
    
    # Apply final touch-ups
    if random.random() > 0.7:
        # Add slight blur occasionally
        collage = collage.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    return collage


def generate_dataset(source_dir, output_dir, num_collages=300, count_probs=None, image_size=448):
    """
    Generate a dataset of receipt collages with varying receipt counts.
    
    Args:
        source_dir: Directory containing individual receipt images
        output_dir: Directory to save the generated collages
        num_collages: Number of collage images to generate
        count_probs: Probability distribution for number of receipts (0-5)
        image_size: Size of output images
        
    Returns:
        DataFrame with image filenames and receipt counts
    """
    import pandas as pd
    
    # Create output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load receipt images
    receipt_images = load_receipt_images(source_dir)
    
    if not receipt_images:
        raise ValueError(f"No receipt images found in {source_dir}")
    
    # Default distribution if not provided
    if count_probs is None:
        count_probs = [0.3, 0.3, 0.2, 0.1, 0.1, 0.0]  # Probabilities for 0, 1, 2, 3, 4, 5 receipts
    
    # Normalize probabilities to sum to 1
    count_probs = np.array(count_probs)
    count_probs = count_probs / count_probs.sum()
    
    # Generate collages
    data = []
    
    for i in range(num_collages):
        # Determine number of receipts based on probability distribution
        receipt_count = np.random.choice(len(count_probs), p=count_probs)
        
        # Create collage
        collage = create_collage(receipt_images, receipt_count, image_size)
        
        # Save image
        filename = f"receipt_collage_{i:05d}.png"
        collage.save(images_dir / filename)
        
        # Add to dataset
        data.append({
            "filename": filename,
            "receipt_count": receipt_count
        })
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_collages} collages")
    
    # Create and save metadata
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    print(f"Dataset generation complete: {num_collages} images")
    print(f"Distribution of receipt counts: {df['receipt_count'].value_counts().sort_index()}")
    
    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate receipt collage dataset")
    parser.add_argument("--source_dir", required=True, help="Directory with source receipt images")
    parser.add_argument("--output_dir", default="receipt_collages", help="Output directory")
    parser.add_argument("--num_collages", type=int, default=300, help="Number of collages to generate")
    parser.add_argument("--count_probs", default="0.3,0.3,0.2,0.1,0.1,0", 
                       help="Probability distribution for receipt counts (0,1,2,3,4,5)")
    parser.add_argument("--image_size", type=int, default=448,
                       help="Size of output images (default: 448 for InternVL2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse probability distribution
    count_probs = [float(p) for p in args.count_probs.split(',')]
    
    # Generate dataset
    df = generate_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_collages=args.num_collages,
        count_probs=count_probs,
        image_size=args.image_size
    )