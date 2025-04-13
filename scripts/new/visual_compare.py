#!/usr/bin/env python3
"""
Visual comparison tool for receipt and tax document generators.

This script generates sample images from both the original and new ab initio
implementations to visually compare the results.
"""
import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# NOTE: This file has been updated to use the new ab initio implementation
# of receipt and tax document generation. The old implementation in
# data.data_generators has been replaced with data.data_generators_new.
# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, parent_dir)

# Import both implementations
try:
    # Original implementation
    # New ab initio implementation
    from data.data_generators_new.receipt_generator import create_receipt as new_receipt
    from data.data_generators_new.receipt_generator import create_receipt as original_receipt
    from data.data_generators_new.receipt_generator import create_tax_document as original_tax_doc
    from data.data_generators_new.tax_document_generator import create_tax_document as new_tax_doc
    
    IMPLEMENTATIONS_LOADED = True
except ImportError as e:
    print(f"Error importing implementations: {e}")
    IMPLEMENTATIONS_LOADED = False


def create_comparison_grid(output_path, num_samples=3, image_size=1024, seed=42):
    """
    Create a visual comparison grid showing original vs. new implementation.
    
    Args:
        output_path: Path to save comparison image
        num_samples: Number of sample images to generate per type
        image_size: Base size for generated images
        seed: Random seed for reproducibility
    """
    if not IMPLEMENTATIONS_LOADED:
        print("Cannot create comparison - implementations not loaded")
        return
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Create a large canvas for the comparison grid
    grid_width = image_size * 2  # Original vs. New
    grid_height = image_size * 2 * num_samples  # Receipts and Tax Docs * samples
    
    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)
    
    # Try to load font
    try:
        font = ImageFont.truetype("Arial", 40)
    except Exception as e:
        print(f"Font 'Arial' not available: {e}")
        try:
            font = ImageFont.truetype("DejaVuSans", 40)
        except Exception as e:
            print(f"Font 'DejaVuSans' not available: {e}")
            font = ImageFont.load_default()
    
    # Add headers
    draw.text((image_size // 2 - 100, 20), "Original Implementation", fill="black", font=font)
    draw.text((image_size + image_size // 2 - 100, 20), "Ab Initio Implementation", fill="black", font=font)
    
    # Generate receipt samples
    for i in range(num_samples):
        # Original receipt
        print(f"Generating original receipt sample {i+1}/{num_samples}")
        orig_receipt = original_receipt(width=random.randint(350, 500), height=random.randint(1000, 1500))
        
        # Resize to fit in grid
        scale_factor = min(image_size / orig_receipt.width, image_size / orig_receipt.height) * 0.8
        new_width = int(orig_receipt.width * scale_factor)
        new_height = int(orig_receipt.height * scale_factor)
        orig_receipt = orig_receipt.resize((new_width, new_height), Image.BICUBIC)
        
        # Position in grid
        x_pos = (image_size - new_width) // 2
        y_pos = (image_size - new_height) // 2 + i * image_size * 2
        
        grid.paste(orig_receipt, (x_pos, y_pos))
        
        # New receipt
        print(f"Generating new receipt sample {i+1}/{num_samples}")
        new_receipt_img = new_receipt(image_size)
        
        # Resize to fit in grid
        scale_factor = min(image_size / new_receipt_img.width, image_size / new_receipt_img.height) * 0.8
        new_width = int(new_receipt_img.width * scale_factor)
        new_height = int(new_receipt_img.height * scale_factor)
        new_receipt_img = new_receipt_img.resize((new_width, new_height), Image.BICUBIC)
        
        # Position in grid
        x_pos = image_size + (image_size - new_width) // 2
        y_pos = (image_size - new_height) // 2 + i * image_size * 2
        
        grid.paste(new_receipt_img, (x_pos, y_pos))
        
        # Add label
        draw.text((20, y_pos + 10), f"Receipt Sample {i+1}", fill="black", font=font)
    
    # Generate tax document samples
    for i in range(num_samples):
        # Original tax document
        print(f"Generating original tax document sample {i+1}/{num_samples}")
        orig_tax = original_tax_doc(image_size=image_size)
        
        # Resize to fit in grid
        scale_factor = 0.8  # Allow some margin
        new_width = int(orig_tax.width * scale_factor)
        new_height = int(orig_tax.height * scale_factor)
        orig_tax = orig_tax.resize((new_width, new_height), Image.BICUBIC)
        
        # Position in grid
        x_pos = (image_size - new_width) // 2
        y_pos = (image_size - new_height) // 2 + i * image_size * 2 + image_size
        
        grid.paste(orig_tax, (x_pos, y_pos))
        
        # New tax document
        print(f"Generating new tax document sample {i+1}/{num_samples}")
        new_tax_img = new_tax_doc(image_size)
        
        # Resize to fit in grid
        scale_factor = 0.8  # Allow some margin
        new_width = int(new_tax_img.width * scale_factor)
        new_height = int(new_tax_img.height * scale_factor)
        new_tax_img = new_tax_img.resize((new_width, new_height), Image.BICUBIC)
        
        # Position in grid
        x_pos = image_size + (image_size - new_width) // 2
        y_pos = (image_size - new_height) // 2 + i * image_size * 2 + image_size
        
        grid.paste(new_tax_img, (x_pos, y_pos))
        
        # Add label
        draw.text((20, y_pos + 10), f"Tax Doc Sample {i+1}", fill="black", font=font)
    
    # Save comparison grid
    print(f"Saving comparison grid to {output_path}")
    grid.save(output_path)
    print(f"Comparison grid saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare receipt generators")
    parser.add_argument("--output", default="implementation_comparison.png", 
                      help="Output file path for comparison image")
    parser.add_argument("--samples", type=int, default=2, 
                      help="Number of samples per document type")
    parser.add_argument("--size", type=int, default=1024,
                      help="Base size for generated images")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create the comparison grid
    create_comparison_grid(
        output_path=args.output,
        num_samples=args.samples,
        image_size=args.size,
        seed=args.seed
    )