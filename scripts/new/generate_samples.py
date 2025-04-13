#!/usr/bin/env python3
"""
Generate sample receipt and tax document images.

This is a simplified script to generate individual sample images
without requiring external dependencies like numpy or pandas.
"""
import os
import sys
from pathlib import Path

# NOTE: This file has been updated to use the new ab initio implementation
# of receipt and tax document generation. The old implementation in
# data.data_generators has been replaced with data.data_generators_new.
# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.insert(0, parent_dir)

# Function to create directory if it doesn't exist
def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    # Create output directory
    output_dir = "sample_images"
    ensure_dir(output_dir)
    
    print(f"Generating sample receipt and tax document images in '{output_dir}'...")
    
    try:
        # Try to import and use our receipt generator
        print("Generating receipt samples...")
        from data.data_generators_new.receipt_generator import create_receipt
        
        # Generate 4 receipt samples (one of each style)
        receipt = create_receipt(image_size=2048)
        receipt.save(f"{output_dir}/receipt_sample1.png")
        print(f"✅ Created {output_dir}/receipt_sample1.png")
        
        receipt = create_receipt(image_size=2048)
        receipt.save(f"{output_dir}/receipt_sample2.png")
        print(f"✅ Created {output_dir}/receipt_sample2.png")
        
    except Exception as e:
        print(f"❌ Error generating receipt samples: {e}")
    
    try:
        # Try to import and use our tax document generator
        print("\nGenerating tax document samples...")
        from data.data_generators_new.tax_document_generator import create_tax_document
        
        # Generate 2 tax document samples
        tax_doc = create_tax_document(image_size=2048)
        tax_doc.save(f"{output_dir}/tax_document_sample1.png")
        print(f"✅ Created {output_dir}/tax_document_sample1.png")
        
        tax_doc = create_tax_document(image_size=2048)
        tax_doc.save(f"{output_dir}/tax_document_sample2.png")
        print(f"✅ Created {output_dir}/tax_document_sample2.png")
        
    except Exception as e:
        print(f"❌ Error generating tax document samples: {e}")
    
    print("\nSample generation complete!")
    print(f"Images saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()