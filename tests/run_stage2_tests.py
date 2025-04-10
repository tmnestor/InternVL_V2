#!/usr/bin/env python3
"""
Runner script for Stage 2 verification tests.

This script runs all the verification tests for Stage 2 (Multimodal Dataset Implementation).
"""
import argparse
import unittest
import sys
from pathlib import Path


def run_tests(verbose=False):
    """Run Stage 2 verification tests."""
    # Configure test loader
    loader = unittest.TestLoader()
    
    # Load tests from test modules
    test_dir = Path(__file__).parent
    
    # Run specific validation tests for Stage 2
    verification_tests = loader.discover(
        str(test_dir / "verification"),
        pattern="test_stage2_*.py",
    )
    
    # Configure test runner
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    
    # Run tests
    result = runner.run(verification_tests)
    
    # Return success status for CI/CD
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 verification tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Display verbose test output")
    args = parser.parse_args()
    
    sys.exit(run_tests(verbose=args.verbose))