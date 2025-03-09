#!/usr/bin/env python3
"""
Index Provider Module

This module abstracts the platform-specific logic for loading and managing the Byaldi index.
It provides a clean interface for getting the index regardless of the underlying platform.
"""

import os
import gc
import logging
import platform
import threading
import fcntl
from pathlib import Path
from typing import Dict, Optional, Any
import pickle
import tempfile
import uuid
import time
import sys

import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Check if running on Apple Silicon
is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the appropriate RAGMultiModalModel based on platform
if is_apple_silicon:
    logger.info("Using custom RAGMultiModalModel optimized for Apple Silicon")
    # On Apple Silicon, use our custom implementation that forces CPU usage with memory-mapped tensors
    from custom_modules.byaldi import RAGMultiModalModel
else:
    logger.info("Using standard RAGMultiModalModel implementation")
    # On other platforms (including GPU systems), use the standard implementation
    from byaldi.RAGModel import RAGMultiModalModel

# Import centralized configuration
from src.config import VISUAL_CONFIG as Config

# Create a persistent cache file path in the system temp directory
# This file will survive hot reloads but will be cleaned up when the system restarts
CACHE_DIR = Path(tempfile.gettempdir()) / "byaldi_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "index_instance_id.pkl"
LOCK_FILE = CACHE_DIR / "index_lock"

class IndexProvider:
    """
    Provides access to the Byaldi index, abstracting away platform-specific details.
    This class is responsible for loading, initializing, and managing the index.
    """
    
    # Class-level variables for shared resources
    _index_instance = None
    _index_lock = threading.RLock()  # Lock for initializing the index within a process
    _index_id = None  # ID to check if we need to reload the index
    
    @classmethod
    def get_index(cls, index_name: str = "visual_books") -> RAGMultiModalModel:
        """
        Get the shared index instance. If it doesn't exist, initialize it.
        
        Args:
            index_name: Name of the Byaldi index
            
        Returns:
            The initialized RAGMultiModalModel instance
        """
        with cls._index_lock:
            # Check if we already have a valid instance
            if cls._index_instance is not None:
                return cls._index_instance
            
            # Use file-based locking to coordinate between processes
            with open(LOCK_FILE, 'w') as lock_file:
                try:
                    # Acquire an exclusive lock
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    
                    # Check again after acquiring the lock (another process might have initialized it)
                    if cls._index_instance is not None:
                        return cls._index_instance
                    
                    # Try to load from cache first
                    if CACHE_FILE.exists():
                        try:
                            with open(CACHE_FILE, 'rb') as f:
                                cached_data = pickle.load(f)
                                cls._index_id = cached_data.get('id')
                                logger.info(f"Found cached index ID: {cls._index_id}")
                        except Exception as e:
                            logger.warning(f"Error loading cache: {e}")
                    
                    # Initialize the index
                    cls._initialize_index(index_name)
                    
                    # Save to cache
                    if cls._index_instance is not None and cls._index_id is not None:
                        try:
                            with open(CACHE_FILE, 'wb') as f:
                                pickle.dump({'id': cls._index_id}, f)
                        except Exception as e:
                            logger.warning(f"Error saving cache: {e}")
                    
                    return cls._index_instance
                finally:
                    # Release the lock
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
    
    @classmethod
    def _initialize_index(cls, index_name: str) -> None:
        """
        Initialize the index with platform-specific optimizations.
        
        Args:
            index_name: Name of the Byaldi index
        """
        try:
            # Check if we already have an instance in memory
            if cls._index_instance is not None:
                logger.info("Index already loaded in memory")
                return
            
            logger.info(f"Starting index initialization with index_name={index_name}")
            
            # Get the index path from environment or config
            index_root = os.getenv("VISUAL_INDEX_PATH", str(Config.INDEX_ROOT))
            logger.info(f"Index root path: {index_root}")
            
            # Get the model name from environment or config
            model_name = os.getenv("VISUAL_MODEL_NAME", Config.MODEL_NAME)
            logger.info(f"Loading model {model_name}")
            
            # Get the device from environment or config
            device = os.getenv("VISUAL_DEVICE", Config.DEVICE)
            logger.info(f"Visual device: {device}  # or cuda if using GPU")
            
            # Log memory usage before loading
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"Memory usage before loading model: {memory_info.rss / 1024 / 1024:.2f} MB")
            
            # Initialize the model with the appropriate settings
            logger.info("Initializing RAGMultiModalModel from index - this may take some time...")
            
            # Generate a unique ID for this instance
            cls._index_id = str(uuid.uuid4())
            
            # Convert the index path to an absolute path
            absolute_index_path = os.path.join(project_root, index_root, index_name)
            logger.info(f"Using absolute index path: {absolute_index_path}")
            
            # Load the index
            if is_apple_silicon:
                # Apple Silicon: Use memory-mapped tensors and force CPU
                cls._index_instance = RAGMultiModalModel.from_index(index_path=absolute_index_path)
            else:
                # Other platforms: Use standard initialization
                cls._index_instance = RAGMultiModalModel.from_index(index_path=absolute_index_path)
            
            # Log success
            logger.info(f"Successfully initialized index with ID {cls._index_id}")
            
        except Exception as e:
            logger.error(f"Error initializing RAGMultiModalModel: {e}")
            raise
    
    @classmethod
    def close_index(cls) -> None:
        """
        Close the index and free resources.
        """
        with cls._index_lock:
            # Use file-based locking to coordinate between processes
            with open(LOCK_FILE, 'w') as lock_file:
                try:
                    # Acquire an exclusive lock
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    
                    if cls._index_instance is not None:
                        logger.info("Closing index and freeing resources")
                        cls._index_instance = None
                        cls._index_id = None
                        
                        # Clear the cache file
                        if CACHE_FILE.exists():
                            try:
                                CACHE_FILE.unlink()
                            except Exception as e:
                                logger.warning(f"Error removing cache file: {e}")
                finally:
                    # Release the lock
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
    
    @classmethod
    def get_doc_ids_to_file_names(cls) -> Dict[int, str]:
        """
        Get a mapping of document IDs to file names.
        
        Returns:
            A dictionary mapping document IDs to file names
        """
        index = cls.get_index()
        if index is None:
            return {}
        
        try:
            return index.get_doc_ids_to_file_names()
        except Exception as e:
            logger.error(f"Error getting doc_ids_to_file_names: {e}")
            return {} 