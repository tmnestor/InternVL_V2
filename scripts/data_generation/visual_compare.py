#!/usr/bin/env python3
"""
Visual comparison tool for receipt and tax document generation approaches.

This script generates sample receipts and tax documents using both the original
and ab initio implementations for visual comparison.
"""
import sys
from pathlib import Path

# Add project root to path to import from scripts/new
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from archive.scripts.new.visual_compare import *


if __name__ == "__main__":
    main()
