#!/usr/bin/env python3
"""
Runner script for Stage 3 verification tests.

This script runs the verification tests for Stage 3 (Training Pipeline),
which focuses on the training infrastructure for both vision-only and
multimodal models.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test module
from tests.verification.test_stage3_verification import TestTrainingPipeline


if __name__ == "__main__":
    # Create a test suite with our test cases
    suite = unittest.TestSuite()
    
    # Add all test methods from the TestTrainingPipeline class
    for method_name in dir(TestTrainingPipeline):
        if method_name.startswith('test_'):
            suite.addTest(TestTrainingPipeline(method_name))
    
    # Run the tests with a more verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with status code based on test results
    sys.exit(not result.wasSuccessful())