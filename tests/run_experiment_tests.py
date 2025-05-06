#!/usr/bin/env python3

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n=== YAML Configuration and Experiment Manager Tests ===\n")

# Import the test class - use absolute import with the full path
from tests.test_experiment_manager import TestExperimentManager

if __name__ == "__main__":
    # Create a test suite with just the experiment manager tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExperimentManager)
    
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