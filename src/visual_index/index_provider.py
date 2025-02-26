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

class IndexProvider:
    """
    Provides access to the Byaldi index, abstracting away platform-specific details.
    This class is responsible for loading, initializing, and managing the index.
    """
    
    # Class-level variables for shared resources
    _index_instance = None
    _index_lock = threading.RLock()  # Lock for initializing the index
    
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
            if cls._index_instance is None:
                cls._initialize_index(index_name)
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
                del cls._index_instance
                cls._index_instance = None
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