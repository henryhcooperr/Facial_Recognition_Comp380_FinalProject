#!/usr/bin/env python3

import os
import sys
import unittest

# Add parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    # Parse command line args
    args = sys.argv[1:]
    
    # Default is to run everything - this is what I use most
    testPattern = "test_tracking*.py"
    
    # Let user pick unit or integration tests only
    if args and args[0] in ["unit", "integration"]:
        if args[0] == "unit":
            # Just run the basic unit tests - way faster
            testPattern = "test_tracking.py"
            print("Running unit tests only - much faster!")
        elif args[0] == "integration":
            # Just run integration tests - good when fixing integration issues
            testPattern = "test_tracking_integration.py"
            print("Running integration tests only - these test the full system")
        
        # Get rid of the first arg
        args = args[1:]
    else:
        print("Running ALL tracking tests - this will take a bit...")
    
    # Find the tests
    testDir = os.path.dirname(__file__)
    finder = unittest.TestLoader()
    testSuite = finder.discover(
        start_dir=testDir,
        pattern=testPattern,
    )
    
    # Run 'em
    testRunner = unittest.TextTestRunner(verbosity=2)
    testResult = testRunner.run(testSuite)
    
    # Exit with proper code (0 = success, 1 = failure)
    # This is important for CI/CD integration
    sys.exit(not testResult.wasSuccessful()) 