#!/usr/bin/env python3

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n=== Training Enhancements Tests ===\n")

# Import the test class
from tests.test_training_utils import TestTrainingUtils

if __name__ == "__main__":
    # Create a test suite with the training utils tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainingUtils)
    
    # Run the tests with verbose output
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Print summary
    print(f"\n=== Test Results ===")
    print(f"Ran {result.testsRun} tests")
    print(f"Success: {result.testsRun - len(result.errors) - len(result.failures)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Exit with proper exit code
    sys.exit(len(result.failures) + len(result.errors)) 