#!/usr/bin/env python3

"""
Test Runner for Face Recognition System
---------------------------------------

This script runs all the tests for the face recognition system.
"""

import unittest
import sys
from pathlib import Path

def run_tests():
    """Discover and run all tests."""
    # Add the project root to the path
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests()) 