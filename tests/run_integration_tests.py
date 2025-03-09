#!/usr/bin/env python
"""
Script to run the integration tests.

This script runs the integration tests against the running services.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the integration tests."""
    # Get the directory of this script
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Get the project root directory
    project_root = script_dir.parent
    
    # Add the project root to the Python path
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    # Run the tests
    print("Running integration tests...")
    
    # Test the services
    print("\n=== Testing Services ===")
    service_result = subprocess.run(
        ["pytest", "-xvs", "integration/test_services.py"],
        cwd=script_dir
    )
    
    # Test ChromaDB initialization
    print("\n=== Testing ChromaDB Initialization ===")
    chroma_result = subprocess.run(
        ["pytest", "-xvs", "integration/test_chroma_init.py"],
        cwd=script_dir
    )
    
    # Check the results
    if service_result.returncode == 0 and chroma_result.returncode == 0:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed:")
        if service_result.returncode != 0:
            print(f"- Service tests failed with exit code {service_result.returncode}")
        if chroma_result.returncode != 0:
            print(f"- ChromaDB initialization tests failed with exit code {chroma_result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 