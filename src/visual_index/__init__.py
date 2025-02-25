"""
Visual Index Module - Initialization

This module sets critical environment variables before any other imports
to ensure proper operation on Apple Silicon hardware.
"""

import os

# Disable CUDA-related features since Apple Silicon doesn't support CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Disable FlashAttention for CPU usage
os.environ["USE_FLASH_ATTENTION"] = "0"

# Disable tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import the search module
from .search import VisualSearch 