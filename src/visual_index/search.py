#!/usr/bin/env python3
"""
Visual search implementation:
  - Uses CPU for compatibility with Apple Silicon
  - Memory-mapped tensor loading for efficiency
  - Batch PDF-to-image conversion
"""

import os
import gc
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator, Union, Tuple, Any
import argparse

from pdf2image import convert_from_path, pdfinfo_from_path
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import track, Progress, SpinnerColumn, TimeElapsedColumn

from PIL import Image
from pydantic import BaseModel

# Import our custom RAGMultiModalModel instead of the original
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add project root to path
from custom_modules.byaldi import RAGMultiModalModel
from byaldi.objects import Result

# Get workspace root path
WORKSPACE_ROOT = Path(os.getenv('WORKSPACE_ROOT', '/Users/patrickcremin/repo/chat'))

# Ensure index directory exists
INDEX_ROOT = WORKSPACE_ROOT / '.byaldi'
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


# Configure rich logging
logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class Config:
    """Centralized configuration settings."""
    DPI = 300
    CHUNK_SIZE = 10
    MODEL_NAME = "vidore/colqwen2-v1.0"
    DEVICE = "cpu"
    IMAGE_FORMAT = "png"
    INDEX_ROOT = INDEX_ROOT
    FILTER_MULTIPLIER = 3  # Get 3x results when filtering to ensure enough after filtering

class VisualSearch:
    """Vectorizes PDFs (visually) using Byaldi, storing indexes on disk."""
    
    def __init__(self, index_name: str = "visual_books"):
        """Initialize the vectorizer with configurable paths.
        
        Args:
            pdf_dir: Directory containing PDFs and metadata
            index_name: Name of the Byaldi index
        """
        self.index_name = 'visual_books'
        self.index_root = Config.INDEX_ROOT
        self.index_path = self.index_root / self.index_name

        # Use class-level singleton pattern to ensure one index instance
        if not hasattr(VisualSearch, '_index_instance'):
            if self.index_root.exists():
                VisualSearch._index_instance = RAGMultiModalModel.from_index(
                    index_path=self.index_name)
            else:
                raise FileNotFoundError(f"Index directory {self.index_root} does not exist. Please create it first.")
        
        self.RAG = VisualSearch._index_instance
        self.doc_ids_to_file_names = self.RAG.get_doc_ids_to_file_names()

    def search(
        self,
        query: str,
        k: int = 3,
        metadata_filters: Optional[Dict[str, Any]] = None,
        metadata_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None,
        metadata_contains: Optional[Dict[str, str]] = None
    ) -> tuple[List[Result], List[Path]]:
        """Search the indexed PDFs for a given query with optional metadata filtering.
        
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
        if not self.doc_ids_to_file_names:  # Move to __init__
            self.doc_ids_to_file_names = self.RAG.get_doc_ids_to_file_names()
        if not self.doc_ids_to_file_names:
            raise ValueError("No documents indexed")
            
        # Get more results than needed to allow for filtering
        has_filters = any([metadata_filters, metadata_ranges, metadata_contains])
        extra_k = k * Config.FILTER_MULTIPLIER if has_filters else k
        results: Union[List[Result], List[List[Result]]] = self.RAG.search(query, k=extra_k)
        
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
        
        # Match results to filenames
        if isinstance(results[0], list):
            files = [[self.doc_ids_to_file_names[r.doc_id] for r in result_list] 
                    for result_list in results]
        else:
            files = [self.doc_ids_to_file_names[r.doc_id] for r in results]
        
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
            
        filtered = []
        for result in results:
            # Get metadata from the first (and only) dict in the list
            metadata = self.RAG.model.doc_id_to_metadata[result.doc_id][0]
            
            # Check exact matches
            if metadata_filters and not all(
                metadata.get(field) == value 
                for field, value in metadata_filters.items()
            ):
                continue
                
            # Check ranges
            if metadata_ranges and not all(
                min_val <= metadata.get(field, min_val - 1) <= max_val
                for field, (min_val, max_val) in metadata_ranges.items()
            ):
                continue
                
            # Check contains
            if metadata_contains and not all(
                str(value).lower() in str(metadata.get(field, "")).lower()
                for field, value in metadata_contains.items()
            ):
                continue
                
            filtered.append(result)
            
        return filtered

    def close(self):
        """Properly close and cleanup resources."""
        if hasattr(self, 'RAG'):
            del self.RAG
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()