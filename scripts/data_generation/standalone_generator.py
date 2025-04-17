#!/usr/bin/env python3
"""
Standalone receipt and tax document generator with no external dependencies.

This script provides a self-contained implementation for generating synthetic
receipts and tax documents without requiring additional modules.
"""
import sys
from pathlib import Path

# Add project root to path to import from scripts/new
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from archive.scripts.new.standalone_generator import *


if __name__ == "__main__":
    main()
