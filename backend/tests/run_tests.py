#!/usr/bin/env python3
"""
Test runner for RAG system tests
Runs all tests and provides detailed output
"""

import os
import sys
import unittest
from io import StringIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """Run all test suites and provide detailed results"""

    # Discover and load tests
    test_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern="test_*.py")

    # Run tests with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, failfast=False)

    print("=" * 70)
    print("RUNNING RAG SYSTEM TESTS")
    print("=" * 70)

    result = runner.run(suite)

    # Print results
    output = stream.getvalue()
    print(output)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    # Print details of failures and errors
    if result.failures:
        print(f"\n{len(result.failures)} FAILURES:")
        for i, (test, failure) in enumerate(result.failures, 1):
            print(f"\n{i}. {test}")
            print("-" * 50)
            print(failure)

    if result.errors:
        print(f"\n{len(result.errors)} ERRORS:")
        for i, (test, error) in enumerate(result.errors, 1):
            print(f"\n{i}. {test}")
            print("-" * 50)
            print(error)

    return result.wasSuccessful()


def run_specific_test(test_module, test_class=None, test_method=None):
    """Run a specific test module, class, or method"""
    if test_class and test_method:
        suite = unittest.TestLoader().loadTestsFromName(
            f"{test_module}.{test_class}.{test_method}"
        )
    elif test_class:
        suite = unittest.TestLoader().loadTestsFromName(f"{test_module}.{test_class}")
    else:
        suite = unittest.TestLoader().loadTestsFromName(test_module)

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG system tests")
    parser.add_argument(
        "--module", help="Run specific test module (e.g., test_course_search_tool)"
    )
    parser.add_argument("--class", dest="test_class", help="Run specific test class")
    parser.add_argument("--method", help="Run specific test method")

    args = parser.parse_args()

    if args.module:
        success = run_specific_test(args.module, args.test_class, args.method)
    else:
        success = run_all_tests()

    sys.exit(0 if success else 1)
