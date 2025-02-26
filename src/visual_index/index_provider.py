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
from pathlib import Path
from typing import Dict, Optional, Any
import pickle
import tempfile

# Configure logging
logger = logging.getLogger(__name__)

# Detect system type
is_apple_silicon = platform.system() == 'Darwin' and platform.machine().startswith('arm')

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

class IndexProvider:
    """
    Provides access to the Byaldi index, abstracting away platform-specific details.
    This class is responsible for loading, initializing, and managing the index.
    """
    
    # Class-level variables for shared resources
    _index_instance = None
    _index_lock = threading.RLock()  # Lock for initializing the index
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
            # Check if we have a cached index ID
            if cls._index_id is None:
                # Try to load the index ID from the cache file
                if CACHE_FILE.exists():
                    try:
                        with open(CACHE_FILE, 'rb') as f:
                            cls._index_id = pickle.load(f)
                            logger.info(f"Loaded index ID from cache: {cls._index_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load index ID from cache: {e}")
                        cls._index_id = None
            
            # If we have a valid index ID but no instance, try to get it from the global namespace
            if cls._index_id is not None and cls._index_instance is None:
                try:
                    # Try to get the index from the global namespace using the ID
                    import builtins
                    if hasattr(builtins, f"_byaldi_index_{cls._index_id}"):
                        cls._index_instance = getattr(builtins, f"_byaldi_index_{cls._index_id}")
                        logger.info(f"Retrieved index instance from global namespace with ID: {cls._index_id}")
                except Exception as e:
                    logger.warning(f"Failed to retrieve index from global namespace: {e}")
                    cls._index_id = None
                    cls._index_instance = None
            
            # If we still don't have an index instance, initialize it
            if cls._index_instance is None:
                cls._initialize_index(index_name)
                
                # Store the index in the global namespace with a unique ID
                import uuid
                import builtins
                cls._index_id = str(uuid.uuid4())
                setattr(builtins, f"_byaldi_index_{cls._index_id}", cls._index_instance)
                logger.info(f"Stored index instance in global namespace with ID: {cls._index_id}")
                
                # Save the index ID to the cache file
                try:
                    with open(CACHE_FILE, 'wb') as f:
                        pickle.dump(cls._index_id, f)
                        logger.info(f"Saved index ID to cache: {cls._index_id}")
                except Exception as e:
                    logger.warning(f"Failed to save index ID to cache: {e}")
            
            return cls._index_instance
    
    @classmethod
    def _initialize_index(cls, index_name: str) -> None:
        """
        Initialize the shared index that will be used by all instances.
        
        Args:
            index_name: Name of the Byaldi index
        """
        index_root = Config.INDEX_ROOT
        logger.info(f"Starting index initialization with index_name={index_name}")
        logger.info(f"Index root path: {index_root}")
        
        if not index_root.exists():
            logger.error(f"Index directory {index_root} does not exist")
            raise FileNotFoundError(f"Index directory {index_root} does not exist. Please create it first.")
        
        try:
            logger.info(f"Loading model {Config.MODEL_NAME}")
            logger.info("Initializing RAGMultiModalModel from index - this may take some time...")
            
            # Track memory before loading
            import psutil
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory usage before loading model: {mem_before:.2f} MB")
            
            # Load the model with timing
            import time
            start_time = time.time()
            
            # Load the index - the appropriate implementation will be used based on the platform
            # On Apple Silicon: Uses our custom implementation with CPU and memory-mapped tensors
            # On other platforms: Uses the standard implementation which can use GPU if available
            cls._index_instance = RAGMultiModalModel.from_index(index_path=index_name)
            
            end_time = time.time()
            
            # Track memory after loading
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory usage after loading model: {mem_after:.2f} MB")
            logger.info(f"Memory increase: {mem_after - mem_before:.2f} MB")
            logger.info(f"Model loading took {end_time - start_time:.2f} seconds")
            
            # Log which device is being used (for informational purposes)
            if hasattr(cls._index_instance, 'device'):
                logger.info(f"Model loaded on device: {cls._index_instance.device}")
            elif hasattr(cls._index_instance.model, 'device'):
                logger.info(f"Model loaded on device: {cls._index_instance.model.device}")
            
            logger.info("RAGMultiModalModel initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAGMultiModalModel: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def close_index(cls) -> None:
        """Close the shared index and free resources."""
        with cls._index_lock:
            if cls._index_instance is not None:
                logger.info("Closing shared RAGMultiModalModel")
                
                # Remove from global namespace if it exists there
                if cls._index_id is not None:
                    try:
                        import builtins
                        if hasattr(builtins, f"_byaldi_index_{cls._index_id}"):
                            delattr(builtins, f"_byaldi_index_{cls._index_id}")
                            logger.info(f"Removed index instance from global namespace with ID: {cls._index_id}")
                    except Exception as e:
                        logger.warning(f"Failed to remove index from global namespace: {e}")
                
                # Delete the instance and clear the cache file
                del cls._index_instance
                cls._index_instance = None
                cls._index_id = None
                
                if CACHE_FILE.exists():
                    try:
                        CACHE_FILE.unlink()
                        logger.info("Removed index ID cache file")
                    except Exception as e:
                        logger.warning(f"Failed to remove cache file: {e}")
                
                gc.collect()
                logger.info("Shared RAGMultiModalModel closed successfully")
    
    @classmethod
    def get_doc_ids_to_file_names(cls) -> Dict[int, str]:
        """
        Get the mapping of document IDs to file names.
        
        Returns:
            A dictionary mapping document IDs to file names
        """
        with cls._index_lock:
            if cls._index_instance is None:
                raise RuntimeError("Index not initialized")
            return cls._index_instance.get_doc_ids_to_file_names() 