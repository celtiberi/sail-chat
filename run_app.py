#!/usr/bin/env python3
"""
Wrapper script to run the application with the correct Python path.
This ensures that all imports work correctly regardless of how the script is invoked.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now run the app
from src.app import main

if __name__ == "__main__":
    main() 