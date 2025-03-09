#!/usr/bin/env python
"""
Client for the Combined Service

This client provides methods to interact with the combined service
for visual search and ChromaDB.
"""

import os
import logging
import httpx
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel
from pathlib import Path
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SearchResult(BaseModel):
    """Result from a visual search query."""
    doc_id: int
    file_name: str
    score: float
    metadata: Dict[str, Any] = {}

class ServiceClient:
    """
    Client for the Combined Service.
    
    This client provides methods to interact with the combined service
    for visual search and ChromaDB.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize the client with the service URL.
        
        Args:
            base_url: The base URL of the service
        """
        self.base_url = base_url or os.getenv("SERVICES_URL", "http://localhost:8081")
        logger.info(f"Initializing ServiceClient with base URL: {self.base_url}")
        self.client = None
    
    async def _get_client(self):
        """Get or create an HTTP client."""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=60.0)
        return self.client
    
    async def close(self):
        """Close the HTTP client connection."""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("Closed HTTP client connection")
    
    async def visual_search(
        self,
        query: str,
        k: int = 3,
        metadata_filters: Optional[Dict[str, Any]] = None,
        metadata_ranges: Optional[Dict[str, Any]] = None,
        metadata_contains: Optional[Dict[str, str]] = None
    ) -> Tuple[List[SearchResult], List[str]]:
        """
        Search the visual index for the given query.
        
        Args:
            query: The search query
            k: Number of results to return
            metadata_filters: Optional filters for metadata fields
            metadata_ranges: Optional range filters for metadata fields
            metadata_contains: Optional contains filters for metadata fields
            
        Returns:
            A tuple of (results, paths) where results is a list of SearchResult objects
            and paths is a list of paths to the result files as strings
        """
        try:
            # Prepare the request payload
            payload = {
                "query": query,
                "k": k,
                "metadata_filters": metadata_filters,
                "metadata_ranges": metadata_ranges,
                "metadata_contains": metadata_contains
            }
            
            # Make the request to the service
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/visual/search",
                json=payload
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Convert the results to SearchResult objects
            results = [SearchResult(**result) for result in data["results"]]
            
            # Create string paths from the file names
            paths = [result.file_name for result in results]
            
            # Log the query time
            logger.info(f"Visual search query completed in {data['query_time_ms']:.2f} ms")
            
            return results, paths
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during visual search: {e.response.text}")
            raise RuntimeError(f"Visual search service returned error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error during visual search: {e}")
            raise RuntimeError(f"Could not connect to visual search service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during visual search: {e}")
            raise RuntimeError(f"Visual search failed: {e}")
    
    async def chroma_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search ChromaDB for the given query.
        
        Args:
            query: The search query
            k: Number of results to return
            filter: Optional filter for metadata fields
            
        Returns:
            A list of Document objects
        """
        try:
            # Prepare the request payload
            payload = {
                "query": query,
                "k": k,
                "filter": filter
            }
            
            # Make the request to the service
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/chroma/search",
                json=payload
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Convert the results to Document objects
            documents = []
            for result in data["results"]:
                doc = Document(
                    page_content=result["page_content"],
                    metadata=result["metadata"]
                )
                documents.append(doc)
            
            # Log the query time
            logger.info(f"ChromaDB search query completed in {data['query_time_ms']:.2f} ms")
            
            return documents
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during ChromaDB search: {e.response.text}")
            raise RuntimeError(f"ChromaDB service returned error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error during ChromaDB search: {e}")
            raise RuntimeError(f"Could not connect to ChromaDB service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during ChromaDB search: {e}")
            raise RuntimeError(f"ChromaDB search failed: {e}")
    
    async def health_check(self) -> bool:
        """
        Check if the service is healthy.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            
            # Check if the request was successful
            if response.status_code != 200:
                logger.warning(f"Health check failed with status code: {response.status_code}")
                return False
            
            # Parse the response
            data = response.json()
            
            # Check if the service is healthy
            if data.get("status") != "ok":
                logger.warning(f"Health check failed: {data.get('message')}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return False 