#!/usr/bin/env python3
"""
Test runner for Stage 1 tests.

This script runs all the tests for Stage 1 of the InternVL Receipt Counter project,
which focuses on the model architecture extension for multimodal capabilities.

Usage:
    python run_stage1_tests.py [-v] [--failfast] [test_name]

Arguments:
    -v, --verbose     : Increase output verbosity
    --failfast        : Stop the test run on the first error or failure
    test_name         : Optional test name pattern (e.g., 'TestCrossAttention')
"""

import argparse
import os
import sys
import unittest


def get_test_modules():
    """Return the list of test modules for Stage 1."""
    return [
        'test_stage1_components',
        'test_stage1_model_architecture',
        'test_stage1_integration'
    ]


def run_tests(verbosity=1, failfast=False, pattern=None):
    """
    Run all Stage 1 tests with the specified options.
    
    Args:
        verbosity: Level of test output verbosity (1=default, 2=verbose)
        failfast: Whether to stop on first failure/error
        pattern: Optional pattern to filter test names
    
    Returns:
        TestResult object with test results
    """
    # Add the parent directory to the path to allow importing test modules
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Get all test modules
    modules = get_test_modules()
    
    # Import each module and add its tests to the suite
    for module_name in modules:
        # Try to import the module
        try:
            module = __import__(module_name)
            
            # If pattern is specified, filter tests by name
            if pattern:
                # Load tests that match the pattern
                tests = unittest.defaultTestLoader.loadTestsFromName(pattern, module)
            else:
                # Load all tests from the module
                tests = unittest.defaultTestLoader.loadTestsFromModule(module)
            
            test_suite.addTests(tests)
        except ImportError as e:
            print(f"Warning: Could not import test module '{module_name}': {e}")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=failfast)
    return runner.run(test_suite)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Stage 1 tests for InternVL Receipt Counter")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("--failfast", action="store_true", help="Stop on first fail or error")
    parser.add_argument("pattern", nargs="?", help="Optional test name pattern")
    
    args = parser.parse_args()
    
    # Set verbosity level
    verbosity = 2 if args.verbose else 1
    
    # Run tests
    result = run_tests(verbosity=verbosity, failfast=args.failfast, pattern=args.pattern)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())