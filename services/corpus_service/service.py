#!/usr/bin/env python
"""
Combined Service for Visual Index and ChromaDB

This service provides a REST API for both the visual index and ChromaDB.
It loads both resources once and keeps them in memory, serving requests
through a unified API.
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the visual search components
from src.visual_index.search import VisualSearch
from src.visual_index.index_provider import IndexProvider

# Import the ChromaDB components
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CONFIG

# Create the FastAPI app
app = FastAPI(
    title="Nautical Assistant Services",
    description="A service for visual search and ChromaDB using a unified API",
    version="1.0.0"
)

# Global variables for shared resources
visual_search = None
chroma_db = None

# Define the request and response models for visual search
class VisualSearchRequest(BaseModel):
    query: str
    k: int = 3
    metadata_filters: Optional[Dict[str, Any]] = None
    metadata_ranges: Optional[Dict[str, Any]] = None
    metadata_contains: Optional[Dict[str, str]] = None

class SearchResult(BaseModel):
    doc_id: int
    file_name: str
    score: float
    metadata: Dict[str, Any] = {}

class VisualSearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ms: float

# Define the request and response models for ChromaDB
class ChromaSearchRequest(BaseModel):
    query: str
    k: int = 5
    filter: Optional[Dict[str, Any]] = None

class ChromaDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}

class ChromaSearchResponse(BaseModel):
    results: List[ChromaDocument]
    query_time_ms: float

# Initialize the shared resources on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the shared resources when the service starts."""
    global visual_search, chroma_db
    
    logger.info("Initializing shared resources...")
    
    try:
        # Initialize the visual search index
        logger.info("Initializing visual search index...")
        VisualSearch.initialize_shared_index()
        visual_search = VisualSearch()
        logger.info("Visual search index initialized successfully")
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        embeddings = HuggingFaceEmbeddings(model_name=CONFIG.embedding_model)
        chroma_db = Chroma(
            collection_name="forum_content",
            embedding_function=embeddings,
            persist_directory=str(CONFIG.chroma_db_dir)
        )
        logger.info(f"ChromaDB initialized with {chroma_db._collection.count()} documents")
        
        logger.info("All shared resources initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing shared resources: {e}")
        raise

# Clean up resources on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the service shuts down."""
    global visual_search, chroma_db
    
    logger.info("Cleaning up resources...")
    
    try:
        # Clean up the visual search index
        if visual_search:
            logger.info("Cleaning up visual search index...")
            VisualSearch.cleanup_shared_index()
            logger.info("Visual search index cleaned up successfully")
        
        # Clean up ChromaDB
        if chroma_db:
            logger.info("Cleaning up ChromaDB...")
            # No explicit cleanup needed for ChromaDB
            logger.info("ChromaDB cleaned up successfully")
        
        logger.info("All resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up resources: {e}")

# Define the visual search endpoint
@app.post("/visual/search", response_model=VisualSearchResponse)
async def visual_search_endpoint(request: VisualSearchRequest):
    """
    Search the visual index for the given query.
    
    Args:
        request: The search request containing the query and optional filters
        
    Returns:
        A list of search results with scores and metadata
    """
    global visual_search
    
    try:
        # Perform the search
        start_time = time.time()
        results, paths = visual_search.search(
            query=request.query,
            k=request.k,
            metadata_filters=request.metadata_filters,
            metadata_ranges=request.metadata_ranges,
            metadata_contains=request.metadata_contains
        )
        end_time = time.time()
        
        # Convert the results to the response format
        search_results = []
        for i, result in enumerate(results):
            # Get the file name from the paths list
            file_name = str(paths[i]).split("pdfs/", 1)[1] if i < len(paths) else "unknown"
            
            # Get metadata from the result object
            metadata = {}
            if hasattr(result, 'metadata') and result.metadata:
                # Log the metadata type and content for debugging
                logger.debug(f"Result {i} metadata type: {type(result.metadata)}")
                logger.debug(f"Result {i} metadata content: {result.metadata}")
                
                # Handle different metadata formats
                if isinstance(result.metadata, list):
                    if len(result.metadata) > 0:
                        if isinstance(result.metadata[0], dict):
                            # If it's a list of dicts, use the first dict
                            metadata = result.metadata[0]
                        else:
                            # If it's a list of non-dicts, create a dict with the list as a value
                            metadata = {"items": result.metadata}
                    else:
                        # Empty list case
                        metadata = {}
                elif isinstance(result.metadata, dict):
                    # If it's already a dict, use it directly
                    metadata = result.metadata
                else:
                    # For any other type, convert to a dict with a single key
                    metadata = {"value": str(result.metadata)}
            
            try:
                # Create a SearchResult object
                search_results.append(SearchResult(
                    doc_id=result.doc_id,
                    file_name=file_name,
                    score=result.score,
                    metadata=metadata
                ))
            except Exception as e:
                # Log the error but continue processing other results
                logger.error(f"Error creating SearchResult for result {i}: {e}")
                logger.error(f"Problematic metadata: {metadata}")
        
        # Calculate the query time in milliseconds
        query_time_ms = (end_time - start_time) * 1000
        
        # Return the response
        return VisualSearchResponse(
            results=search_results,
            query_time_ms=query_time_ms
        )
    except Exception as e:
        logger.error(f"Error performing visual search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Define the ChromaDB search endpoint
@app.post("/chroma/search", response_model=ChromaSearchResponse)
async def chroma_search_endpoint(request: ChromaSearchRequest):
    """
    Search ChromaDB for the given query.
    
    Args:
        request: The search request containing the query and optional filters
        
    Returns:
        A list of documents with metadata
    """
    global chroma_db
    
    try:
        # Log the request details
        logger.info(f"ChromaDB search request: query='{request.query}', k={request.k}, filter={request.filter}")
        
        # Check if ChromaDB is initialized
        if chroma_db is None:
            logger.error("ChromaDB not initialized")
            raise HTTPException(status_code=500, detail="ChromaDB not initialized")
        
        # Log the collection count
        collection_count = chroma_db._collection.count()
        logger.info(f"ChromaDB collection count: {collection_count}")
        
        # Perform the search
        start_time = time.time()
        
        # If the collection is empty, return an empty response
        if collection_count == 0:
            logger.warning("ChromaDB collection is empty")
            return ChromaSearchResponse(
                results=[],
                query_time_ms=0
            )
        
        # Perform the search
        results = chroma_db.similarity_search(
            query=request.query,
            k=request.k,
            filter=request.filter
        )
        end_time = time.time()
        
        # Log the results
        logger.info(f"ChromaDB search returned {len(results)} results")
        if results:
            # Log the first result
            logger.info(f"First result metadata: {results[0].metadata}")
        else:
            logger.warning(f"No results found for query: '{request.query}', filter: {request.filter}")
            
            # If no results with filter, try without filter
            if request.filter:
                logger.info("Trying search without filter")
                results_no_filter = chroma_db.similarity_search(
                    query=request.query,
                    k=request.k
                )
                logger.info(f"Search without filter returned {len(results_no_filter)} results")
                if results_no_filter:
                    logger.info(f"First result metadata (no filter): {results_no_filter[0].metadata}")
        
        # Convert the results to the response format
        chroma_results = []
        for doc in results:
            chroma_results.append(ChromaDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            ))
        
        # Calculate the query time in milliseconds
        query_time_ms = (end_time - start_time) * 1000
        logger.info(f"ChromaDB search query completed in {query_time_ms:.2f} ms")
        
        # Return the response
        return ChromaSearchResponse(
            results=chroma_results,
            query_time_ms=query_time_ms
        )
    except Exception as e:
        logger.error(f"Error performing ChromaDB search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Define a health check endpoint
@app.get("/health")
async def health_check():
    """Check if the service is healthy."""
    global visual_search, chroma_db
    
    try:
        # Check if the visual index is loaded
        visual_index = IndexProvider.get_index()
        if visual_index is None:
            return {"status": "error", "message": "Visual index not loaded"}
        
        # Check if ChromaDB is loaded
        if chroma_db is None:
            return {"status": "error", "message": "ChromaDB not loaded"}
        
        # Return a success response
        return {
            "status": "ok",
            "message": "Service is healthy",
            "visual_index": "loaded",
            "chroma_db": "loaded",
            "chroma_documents": chroma_db._collection.count()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e)}

# Run the service if this file is executed directly
if __name__ == "__main__":
    # Get the port from the environment or use a default
    port = int(os.getenv("SERVICES_PORT", 8081))
    
    # Run the service
    logger.info(f"Starting service on port {port}")
    uvicorn.run("corpus_service.service:app", host="0.0.0.0", port=port, app_dir="services") 