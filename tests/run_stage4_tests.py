#!/usr/bin/env python3
"""
Runner script for Stage 4 verification tests.

This script runs the verification tests for Stage 4 (Training Orchestration and Evaluation),
which focuses on training orchestration, monitoring, and evaluation for the multimodal
vision-language model.
"""

import os
import sys
import unittest
import argparse
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test module
from tests.verification.test_stage4_verification import TestStage4Verification


def main():
    """Run the Stage 4 verification tests."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Stage 4 verification tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--test", type=str, help="Run only the specified test method")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create test suite
    loader = unittest.TestLoader()
    
    if args.test:
        # Run specific test
        suite = loader.loadTestsFromName(args.test, TestStage4Verification)
    else:
        # Run all tests
        suite = loader.loadTestsFromTestCase(TestStage4Verification)
    
    # Set verbosity
    verbosity = 2 if args.verbose else 1
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return exit code based on test results
    return not result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(main())