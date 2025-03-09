"""
Integration tests for the Combined Services.

These tests assume that the services are already running.
"""

import os
import sys
import pytest
import json
from pathlib import Path

# Import the service client
from services.client import ServiceClient

@pytest.mark.asyncio
async def test_health_check(client):
    """Test the health check endpoint."""
    is_healthy = await client.health_check()
    assert is_healthy, "Health check failed"

@pytest.mark.asyncio
async def test_visual_search(client):
    """Test the visual search endpoint."""
    # Test with a simple query
    query = "sailing boat"
    results, paths = await client.visual_search(query=query, k=3)
    
    # Log the results for debugging
    print(f"Visual search returned {len(results)} results")
    if results:
        result = results[0]
        print(f"First result: doc_id={result.doc_id}, file_name={result.file_name}, score={result.score}")
        print(f"Metadata: {json.dumps(result.metadata, default=str)}")
    
    # Assert that each result has a valid file name and the file exists in the data/pdfs directory
    for result in results:
        assert result.file_name, "Result missing file_name value"
        assert len(result.file_name) > 0, "Result has empty file_name"
        # Check that the file exists in the data/pdfs directory
        file_path = Path(os.getenv("DATA_DIR")) / "pdfs" / result.file_name
        assert file_path.exists(), f"File {file_path} does not exist"
        
    # Assert that we got some results
    assert len(results) > 0, "Visual search returned no results"
    assert len(paths) > 0, "Visual search returned no paths"
    
    # Check the structure of the results
    for result in results:
        assert hasattr(result, 'doc_id'), "Result missing doc_id"
        assert hasattr(result, 'file_name'), "Result missing file_name"
        assert hasattr(result, 'score'), "Result missing score"
        assert hasattr(result, 'metadata'), "Result missing metadata"

@pytest.mark.asyncio
async def test_chroma_search(client):
    """Test the ChromaDB search endpoint."""
    # Test with a simple query
    query = "sailing"
    docs = await client.chroma_search(query=query, k=5)
    
    # Log the results for debugging
    print(f"ChromaDB search returned {len(docs)} documents")
    if docs:
        doc = docs[0]
        print(f"First document: {doc.page_content[:100]}...")
        print(f"Metadata: {json.dumps(doc.metadata, default=str)}")
    
    # Assert that we got some results
    assert len(docs) > 0, "ChromaDB search returned no documents"
    
    # Check the structure of the documents
    for doc in docs:
        assert hasattr(doc, 'page_content'), "Document missing page_content"
        assert hasattr(doc, 'metadata'), "Document missing metadata"

@pytest.mark.asyncio
async def test_chroma_search_alternative(client):
    """Test the ChromaDB search endpoint with an alternative query."""
    # Test with a different query
    query = "boat"
    docs = await client.chroma_search(query=query, k=5)
    
    # Log the results for debugging
    print(f"ChromaDB search (alternative) returned {len(docs)} documents")
    if docs:
        doc = docs[0]
        print(f"First document: {doc.page_content[:100]}...")
        print(f"Metadata: {json.dumps(doc.metadata, default=str)}")
    
    # Assert that we got some results
    assert len(docs) > 0, "ChromaDB search returned no documents for alternative query" 