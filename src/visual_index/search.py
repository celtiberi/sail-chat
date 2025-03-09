#!/usr/bin/env python3
"""
Visual search implementation:
  - Uses the IndexProvider to abstract platform-specific details
  - Provides a clean interface for searching visual content
"""

import os
import gc
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator, Union, Tuple, Any
import argparse
import threading
import numpy as np

from pdf2image import convert_from_path, pdfinfo_from_path
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import track, Progress, SpinnerColumn, TimeElapsedColumn

from PIL import Image
from pydantic import BaseModel

# Import the IndexProvider which handles platform-specific details
from .index_provider import IndexProvider

# Import Result class - IndexProvider will handle the platform-specific implementation
from byaldi.objects import Result

# Import centralized configuration
from src.config import VISUAL_CONFIG as Config, WORKSPACE_ROOT

# Ensure index directory exists
INDEX_ROOT = Config.INDEX_ROOT
INDEX_ROOT.mkdir(parents=True, exist_ok=True)

class PDFMetadata(BaseModel):
    """Metadata for a single PDF document."""
    id: int
    title: str
    file: str
    metadata: Dict = {}  # Optional with empty dict default
    processed: bool
    images_processed: int = 0  # Track number of images processed


class PDFCollection(BaseModel):
    """Collection of PDF documents with metadata."""
    pdfs: List[PDFMetadata]


# Configure rich logging - but only if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "DEBUG"),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG"))

# Add a direct print to verify this module is loaded
print(f"Visual search module loaded with log level: {logger.level}")

class VisualSearch:
    """Vectorizes PDFs (visually) using Byaldi, storing indexes on disk."""
    
    def __init__(self, index_name: str = "visual_books"):
        """Initialize a new VisualSearch instance that uses the shared index.
        
        Args:
            index_name: Name of the Byaldi index
        """
        self.index_name = index_name
        self.index_root = Config.INDEX_ROOT
        
        # Get the RAG model from the IndexProvider
        self.RAG = IndexProvider.get_index(index_name)
        
        # Get the document IDs to file names mapping
        self.doc_ids_to_file_names = IndexProvider.get_doc_ids_to_file_names()

    def search(
        self,
        query: str,
        k: int = 3,
        metadata_filters: Optional[Dict[str, Any]] = None,
        metadata_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
        metadata_contains: Optional[Dict[str, str]] = None
    ) -> tuple[List[Result], List[Path]]:
        """Search the indexed PDFs for a given query with optional metadata filtering.
        
        This method is thread-safe and can be called concurrently from multiple instances.
        
        Args:
            query: Search query string
            k: Number of results to return
            metadata_filters: Dict of field:value pairs for exact matches
            metadata_ranges: Dict of field:(min,max) pairs for range matches
            metadata_contains: Dict of field:value pairs for substring matches
        
        Returns:
            Tuple containing list of Result objects and corresponding filenames
        """
        if k < 1:
            raise ValueError("k must be positive")
            
        if not self.doc_ids_to_file_names:
            self.doc_ids_to_file_names = IndexProvider.get_doc_ids_to_file_names()
        if not self.doc_ids_to_file_names:
            raise ValueError("No documents indexed")
            
        # Get more results than needed to allow for filtering
        has_filters = any([metadata_filters, metadata_ranges, metadata_contains])
        extra_k = k * Config.FILTER_MULTIPLIER if has_filters else k
        logger.debug(f"Performing search with query: {query}, k={extra_k}")
        
        # The search operation itself is thread-safe as it's read-only
        results: Union[List[Result], List[List[Result]]] = self.RAG.search(query, k=extra_k)
        
        # Log the raw results structure
        logger.debug(f"Raw search results type: {type(results)}")
        if results:
            logger.debug(f"First result type: {type(results[0])}")
            if hasattr(results[0], 'metadata'):
                logger.debug(f"First result metadata type: {type(results[0].metadata)}")
                logger.debug(f"First result metadata: {results[0].metadata}")
        
        # Handle nested results
        if isinstance(results[0], list):
            filtered_results = []
            for result_group in results:
                filtered_group = self._filter_results(
                    result_group,
                    metadata_filters,
                    metadata_ranges,
                    metadata_contains
                )[:k]  # Trim to k after filtering
                if filtered_group:  # Only add groups that have results after filtering
                    filtered_results.append(filtered_group)
            results = filtered_results
        else:
            results = self._filter_results(
                results,
                metadata_filters,
                metadata_ranges,
                metadata_contains
            )[:k]  # Trim to k after filtering
        
        # Log the filtered results
        logger.debug(f"Filtered results count: {len(results)}")
        if results:
            if isinstance(results[0], list):
                logger.debug(f"First filtered result group size: {len(results[0])}")
                if results[0]:
                    first_result = results[0][0]
                    logger.debug(f"First result in first group: doc_id={first_result.doc_id}, score={first_result.score}")
                    if hasattr(first_result, 'metadata'):
                        logger.debug(f"First result metadata type: {type(first_result.metadata)}")
                        logger.debug(f"First result metadata: {first_result.metadata}")
            else:
                first_result = results[0]
                logger.debug(f"First filtered result: doc_id={first_result.doc_id}, score={first_result.score}")
                if hasattr(first_result, 'metadata'):
                    logger.debug(f"First result metadata type: {type(first_result.metadata)}")
                    logger.debug(f"First result metadata: {first_result.metadata}")
        
        # Match results to filenames
        if isinstance(results[0], list):
            files = [[self.doc_ids_to_file_names[r.doc_id] for r in result_list] 
                    for result_list in results]
        else:
            files = [self.doc_ids_to_file_names[r.doc_id] for r in results]
        
        logger.debug(f"Search completed, found {len(results)} results")
        return results, files

    def _filter_results(
        self,
        results: List[Result],
        metadata_filters: Optional[Dict[str, Any]] = None,
        metadata_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
        metadata_contains: Optional[Dict[str, str]] = None
    ) -> List[Result]:
        """Filter results based on metadata criteria.
        
        Args:
            results: List of Result objects to filter
            metadata_filters: Dict of field:value pairs for exact matches
            metadata_ranges: Dict of field:(min,max) pairs for range matches
            metadata_contains: Dict of field:value pairs for substring matches
            
        Returns:
            Filtered list of Result objects
        """
        if not any([metadata_filters, metadata_ranges, metadata_contains]):
            return results
        
        logger.debug(f"Filtering {len(results)} results with filters: {metadata_filters}, ranges: {metadata_ranges}, contains: {metadata_contains}")
            
        filtered = []
        for i, result in enumerate(results):
            # Get metadata from the first (and only) dict in the list
            try:
                metadata = self.RAG.model.doc_id_to_metadata[result.doc_id][0]
                logger.debug(f"Result {i} (doc_id={result.doc_id}) metadata from doc_id_to_metadata: {metadata}")
                
                # Also log the metadata directly from the result object for comparison
                if hasattr(result, 'metadata'):
                    logger.debug(f"Result {i} metadata from result object: {result.metadata}")
                    logger.debug(f"Result {i} metadata type: {type(result.metadata)}")
                
                # Check exact matches
                if metadata_filters:
                    matches = all(
                        metadata.get(field) == value 
                        for field, value in metadata_filters.items()
                    )
                    if not matches:
                        logger.debug(f"Result {i} failed metadata_filters check")
                        continue
                    
                # Check ranges
                if metadata_ranges:
                    matches = all(
                        min_val <= metadata.get(field, min_val - 1) <= max_val
                        for field, (min_val, max_val) in metadata_ranges.items()
                    )
                    if not matches:
                        logger.debug(f"Result {i} failed metadata_ranges check")
                        continue
                    
                # Check contains
                if metadata_contains:
                    matches = all(
                        str(value).lower() in str(metadata.get(field, "")).lower()
                        for field, value in metadata_contains.items()
                    )
                    if not matches:
                        logger.debug(f"Result {i} failed metadata_contains check")
                        continue
                
                # If we get here, the result passed all filters
                logger.debug(f"Result {i} passed all filters")
                
                # Set the metadata on the result object for consistency
                result.metadata = [metadata]
                
                filtered.append(result)
            except Exception as e:
                logger.error(f"Error filtering result {i} (doc_id={result.doc_id}): {e}")
                # Skip this result if there was an error
        
        logger.debug(f"Filtering complete, {len(filtered)} results passed")
        return filtered

    def close(self):
        """Properly close and cleanup instance resources.
        
        Note: This does not close the shared index, only instance-specific resources.
        To close the shared index, use IndexProvider.close_index().
        """
        # Clear instance-specific resources
        self.doc_ids_to_file_names = None
        self.RAG = None  # Remove reference to shared index
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def initialize_shared_index(cls, index_name: str = "visual_books") -> None:
        """
        Initialize the shared index that will be used by all instances.
        
        Args:
            index_name: Name of the Byaldi index
        """
        # This will use the file-based locking in IndexProvider
        # to ensure only one process loads the index
        IndexProvider.get_index(index_name)
        logger.info("VisualSearch shared index initialized")
    
    @classmethod
    def cleanup_shared_index(cls) -> None:
        """
        Clean up the shared index and free resources.
        """
        IndexProvider.close_index()
        logger.info("VisualSearch shared index cleaned up")