#!/usr/bin/env python
"""
Test file for the visual search service.
This file contains tests to verify the behavior of the visual search service,
particularly focusing on metadata handling.
"""

import os
import sys
import json
import logging
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the service app
from services.corpus_service.service import app

# Create a test client
client = TestClient(app)

# Mock the VisualSearch class
class MockResult:
    def __init__(self, doc_id, score, metadata=None):
        self.doc_id = doc_id
        self.score = score
        self.metadata = metadata

# Test cases for different metadata formats
@pytest.mark.parametrize(
    "metadata,expected_status",
    [
        (None, 200),  # No metadata
        ({}, 200),  # Empty dict
        ({"key": "value"}, 200),  # Simple dict
        ([], 200),  # Empty list
        ([{"key": "value"}], 200),  # List with one dict
        ([{"filename": "test.pdf", "description": "test"}], 200),  # List with one dict containing filename and description
        ([{"filename": "test.pdf"}, {"description": "test"}], 200),  # List with multiple dicts
        ("string_metadata", 200),  # String metadata
        (123, 200),  # Numeric metadata
    ],
)
def test_visual_search_with_different_metadata(metadata, expected_status):
    """Test the visual search endpoint with different metadata formats."""
    # Mock the VisualSearch.search method
    with patch("services.service.visual_search") as mock_visual_search:
        # Create mock results
        mock_result = MockResult(doc_id=1, score=0.95, metadata=metadata)
        mock_visual_search.search.return_value = ([mock_result], ["path/to/pdfs/test.pdf"])
        
        # Make the request
        response = client.post(
            "/visual/search",
            json={"query": "test query", "k": 1}
        )
        
        # Log the response
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response body: {response.json()}")
        
        # Check the status code
        assert response.status_code == expected_status
        
        # If successful, check the response structure
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "query_time_ms" in data
            assert len(data["results"]) == 1
            
            # Check the metadata in the response
            result = data["results"][0]
            assert "metadata" in result
            assert isinstance(result["metadata"], dict)
            
            # Log the metadata
            logger.info(f"Original metadata: {metadata}")
            logger.info(f"Response metadata: {result['metadata']}")

# Test with the specific metadata format from the error
def test_visual_search_with_problematic_metadata():
    """Test the visual search endpoint with the problematic metadata format from the error."""
    # Create the problematic metadata
    problematic_metadata = [{'filename': 'The-Annapolis-Book-of-Seamanship.pdf', 'description': 'sailboat handling'}]
    
    # Mock the VisualSearch.search method
    with patch("services.service.visual_search") as mock_visual_search:
        # Create mock results
        mock_result = MockResult(doc_id=1, score=0.95, metadata=problematic_metadata)
        mock_visual_search.search.return_value = ([mock_result], ["path/to/pdfs/The-Annapolis-Book-of-Seamanship.pdf"])
        
        # Make the request
        response = client.post(
            "/visual/search",
            json={"query": "sailboat handling", "k": 1}
        )
        
        # Log the response
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response body: {response.json() if response.status_code == 200 else response.text}")
        
        # Check the status code
        assert response.status_code == 200
        
        # Check the response structure
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        
        # Check the metadata in the response
        result = data["results"][0]
        assert "metadata" in result
        assert isinstance(result["metadata"], dict)
        
        # Log the metadata
        logger.info(f"Response metadata: {result['metadata']}")

if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__]) 