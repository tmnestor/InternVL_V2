#!/usr/bin/env python3
"""
Verification test runner for InternVL Receipt Counter.

This script runs verification tests for the already implemented stages
of the InternVL Receipt Counter project to ensure correctness.

Usage:
    python run_verification.py [-v] [--stage STAGE] [--coverage] [--report]

Arguments:
    -v, --verbose     : Increase output verbosity
    --stage STAGE     : Run tests for a specific stage (1-4)
    --coverage        : Run tests with coverage analysis
    --report          : Generate HTML coverage report
"""

import argparse
import os
import sys
import unittest
import subprocess
from pathlib import Path


def ensure_verification_directory():
    """Ensure the verification directory exists."""
    verification_dir = Path(__file__).parent / "verification"
    verification_dir.mkdir(exist_ok=True)
    
    # Create an __init__.py file if it doesn't exist
    init_file = verification_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()
    
    return verification_dir


def run_tests(stage=None, verbosity=1):
    """
    Run verification tests with the specified options.
    
    Args:
        stage: Optional stage number to test (1-4)
        verbosity: Level of test output verbosity (1=default, 2=verbose)
    
    Returns:
        TestResult object with test results
    """
    # Add the parent directory to the path to allow importing test modules
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Ensure verification directory exists
    ensure_verification_directory()
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Get test files
    verification_dir = Path(__file__).parent / "verification"
    if stage:
        pattern = f"test_stage{stage}_verification.py"
    else:
        pattern = "test_stage*_verification.py"
    
    test_files = list(verification_dir.glob(pattern))
    
    if not test_files:
        print(f"No verification tests found matching pattern: {pattern}")
        return None
    
    # Add tests to suite
    for test_file in test_files:
        module_name = f"tests.verification.{test_file.stem}"
        try:
            # Import the module
            __import__(module_name)
            module = sys.modules[module_name]
            
            # Add all tests from the module
            tests = unittest.defaultTestLoader.loadTestsFromModule(module)
            test_suite.addTests(tests)
            
        except ImportError as e:
            print(f"Warning: Could not import test module '{module_name}': {e}")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(test_suite)


def run_with_coverage(stage=None, report=False):
    """
    Run verification tests with coverage analysis.
    
    Args:
        stage: Optional stage number to test (1-4)
        report: Whether to generate HTML coverage report
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Check if coverage is installed
    try:
        import coverage
    except ImportError:
        print("Error: coverage package not installed. Run 'pip install coverage' first.")
        return 1
    
    # Construct command
    cmd = [sys.executable, "-m", "coverage", "run", "--source=models,training,data,utils"]
    
    # Add current script
    cmd.extend([__file__])
    
    # Add stage if specified
    if stage:
        cmd.extend(["--stage", str(stage)])
    
    # Run coverage
    result = subprocess.run(cmd, env=os.environ.copy())
    
    # Generate report if requested
    if report and result.returncode == 0:
        print("\nGenerating coverage report...")
        subprocess.run([sys.executable, "-m", "coverage", "report", "-m"])
        
        # Generate HTML report
        report_dir = Path(__file__).parent / "coverage_html"
        subprocess.run([
            sys.executable, "-m", "coverage", "html", 
            f"--directory={report_dir}"
        ])
        print(f"HTML coverage report generated in {report_dir}")
    
    return result.returncode


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run verification tests for InternVL Receipt Counter")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="Run tests for a specific stage")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage analysis")
    parser.add_argument("--report", action="store_true", help="Generate HTML coverage report")
    
    args = parser.parse_args()
    
    # Set verbosity level
    verbosity = 2 if args.verbose else 1
    
    # Check if this is being run by coverage
    running_under_coverage = sys.argv[0].endswith('coverage')
    
    if args.coverage and not running_under_coverage:
        # Run with coverage
        sys.exit(run_with_coverage(args.stage, args.report))
    else:
        # Run normally
        result = run_tests(stage=args.stage, verbosity=verbosity)
        if result:
            sys.exit(not result.wasSuccessful())
        else:
            sys.exit(1)