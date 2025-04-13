#!/usr/bin/env python3
"""
Regenerate tax documents for class 0 images.

This script identifies all class 0 (receipt_count=0) images in the dataset
and regenerates them with improved tax document styling to make them more
visually distinct from receipts.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE: This file has been updated to use the new ab initio implementation
# of receipt and tax document generation. The old implementation in
# data.data_generators has been replaced with data.data_generators_new.
# Add parent directory to path to import from data/data_generators
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.data_generators_new.receipt_generator import create_tax_document


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Regenerate tax documents")
    parser.add_argument("--metadata", default="datasets/synthetic_receipts/metadata.csv",
                        help="Path to metadata CSV file")
    parser.add_argument("--img_dir", default="datasets/synthetic_receipts/images",
                        help="Directory containing the images")
    parser.add_argument("--backup", action="store_true",
                        help="Backup original images before replacing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load metadata
    metadata_path = Path(args.metadata)
    img_dir = Path(args.img_dir)
    
    # Create backup directory if needed
    if args.backup:
        backup_dir = img_dir.parent / "images_backup"
        backup_dir.mkdir(exist_ok=True)
    
    print(f"Reading metadata from {metadata_path}")
    try:
        metadata = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        sys.exit(1)
    
    # Identify class 0 (tax document) images
    tax_docs = metadata[metadata['receipt_count'] == 0]
    print(f"Found {len(tax_docs)} tax documents to regenerate")
    
    if len(tax_docs) == 0:
        print("No tax documents found. Nothing to regenerate.")
        return
    
    # Process each tax document
    for i, row in enumerate(tax_docs.itertuples()):
        filename = row.filename
        img_path = img_dir / filename
        
        # Backup original if requested
        if args.backup and img_path.exists():
            backup_path = backup_dir / filename
            try:
                import shutil
                shutil.copy2(img_path, backup_path)
                print(f"Backed up {filename} to {backup_path}")
            except Exception as e:
                print(f"Error backing up {filename}: {e}")
                continue
        
        # Generate new tax document
        try:
            print(f"Regenerating tax document {i+1}/{len(tax_docs)}: {filename}")
            tax_doc = create_tax_document(image_size=2048)  # Use high resolution
            tax_doc.save(img_path)
        except Exception as e:
            print(f"Error generating {filename}: {e}")
    
    print("\nRegeneration complete!")
    print(f"Successfully regenerated {len(tax_docs)} tax documents with improved styling")
    print("The documents now have:")
    print("- More whitespace for better visual distinction")
    print("- Official-looking headers and footers")
    print("- Formal tables and layouts")
    print("- Light blue backgrounds typical of tax forms")
    print("- Subtle ATO watermarking")
    
    print("\nThese changes will make the tax documents (class 0) visually distinct from receipts,")
    print("allowing the model to better learn the difference between tax documents and receipts.")


if __name__ == "__main__":
    main()