#!/usr/bin/env python
"""
Test file for ChromaDB search functionality.
This file contains tests to verify the behavior of the ChromaDB search,
particularly focusing on why we're not getting any documents back.
"""

import os
import sys
import json
import logging
import pytest
import requests
from unittest.mock import patch, MagicMock
import asyncio

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the service client
from services.corpus_service import CorpusClient

# Import the config
from src.config import RETRIEVER_CONFIG as CONFIG

# Test the ChromaDB search directly
def test_chroma_search_direct():
    """Test the ChromaDB search directly using requests."""
    # Get the services URL from environment or use default
    services_url = os.getenv("SERVICES_URL", "http://localhost:8081")
    
    # Create the request payload
    payload = {
        "query": "sailing techniques",
        "k": 10,
        "filter": None  # Try without filter first
    }
    
    # Make the request
    try:
        response = requests.post(f"{services_url}/chroma/search", json=payload)
        
        # Log the response
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response body: {response.json()}")
        
        # Check if we got any results
        data = response.json()
        assert "results" in data
        logger.info(f"Number of results: {len(data['results'])}")
        
        # Log the first result if available
        if data["results"]:
            logger.info(f"First result: {data['results'][0]}")
        else:
            logger.warning("No results returned")
            
    except Exception as e:
        logger.error(f"Error making request: {e}")
        raise

# Test with different filters
@pytest.mark.parametrize(
    "query,filter,expected_min_results",
    [
        ("sailing", None, 1),  # Basic query, no filter
        ("sailing techniques", None, 1),  # More specific query
        ("sailing", {"topics": "General Sailing Forum"}, 0),  # With topic filter
        ("sailing", {"topic": "General Sailing Forum"}, 0),  # With topic filter (singular)
        ("sailing", {"topics": "General"}, 0),  # With partial topic filter
    ],
)
def test_chroma_search_with_filters(query, filter, expected_min_results):
    """Test the ChromaDB search with different filters."""
    # Get the services URL from environment or use default
    services_url = os.getenv("SERVICES_URL", "http://localhost:8081")
    
    # Create the request payload
    payload = {
        "query": query,
        "k": 10,
        "filter": filter
    }
    
    # Make the request
    try:
        response = requests.post(f"{services_url}/chroma/search", json=payload)
        
        # Log the response
        logger.info(f"Query: {query}, Filter: {filter}")
        logger.info(f"Response status: {response.status_code}")
        
        # Check if we got any results
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            num_results = len(data["results"])
            logger.info(f"Number of results: {num_results}")
            
            # Log the first result if available
            if data["results"]:
                logger.info(f"First result metadata: {data['results'][0]['metadata']}")
            else:
                logger.warning("No results returned")
                
            # Check if we got the expected minimum number of results
            assert num_results >= expected_min_results, f"Expected at least {expected_min_results} results, got {num_results}"
        else:
            logger.error(f"Error response: {response.text}")
            assert False, f"Request failed with status {response.status_code}"
            
    except Exception as e:
        logger.error(f"Error making request: {e}")
        raise

@pytest.mark.parametrize(
    "topic",
    [
        "Engines and Propulsion Systems",
        "Construction, Maintenance & Refit",
        "General Sailing Forum",
        None  # No filter
    ],
)
def test_diesel_engine_query(topic):
    """Test the specific query 'diesel engine overheating fix' with different topic filters."""
    # Get the services URL from environment or use default
    services_url = os.getenv("SERVICES_URL", "http://localhost:8081")
    
    # Define the query
    query = "diesel engine overheating fix"
    
    # Create the request payload
    payload = {
        "query": query,
        "k": 10,
        "filter": {"topics": topic} if topic else None
    }
    
    # Make the request
    try:
        print(f"\nTesting query '{query}' with filter {payload['filter']}")
        response = requests.post(f"{services_url}/chroma/search", json=payload)
        
        # Log the response
        print(f"Response status: {response.status_code}")
        
        # Check if we got any results
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            num_results = len(data["results"])
            print(f"Number of results: {num_results}")
            
            # Log the first result if available
            if data["results"]:
                # Count occurrences of each topic
                topic_counts = {}
                for result in data["results"]:
                    result_topic = result["metadata"].get("topics", "Unknown")
                    topic_counts[result_topic] = topic_counts.get(result_topic, 0) + 1
                
                print(f"Topic distribution in results: {topic_counts}")
                
                # Print the first few results
                print("First few results:")
                for i, result in enumerate(data["results"][:3]):
                    print(f"Result {i+1}:")
                    print(f"  Content: {result['page_content'][:100]}...")
                    print(f"  Topics: {result['metadata'].get('topics', 'Unknown')}")
                
                # Check if the topic matches the filter
                if topic:
                    result_topic = data["results"][0]["metadata"].get("topics", "")
                    print(f"Result topic: {result_topic}, Filter topic: {topic}")
                    # We don't assert equality because the filter might not match exactly
            else:
                print(f"No results returned for query '{query}' with filter {payload['filter']}")
        else:
            print(f"Error response: {response.text}")
            assert False, f"Request failed with status {response.status_code}"
            
    except Exception as e:
        print(f"Error making request: {e}")
        raise

# Test the service client
@pytest.mark.asyncio
async def test_service_client():
    """Test the service client's ChromaDB search functionality."""
    # Get the services URL from environment or use default
    services_url = os.getenv("SERVICES_URL", "http://localhost:8081")
    
    # Create the service client
    client = CorpusClient(services_url)
    
    # Test the health check
    try:
        is_healthy = await client.health_check()
        logger.info(f"Service health: {is_healthy}")
        assert is_healthy, "Service is not healthy"
    except Exception as e:
        logger.error(f"Error checking service health: {e}")
        raise
    
    # Test the ChromaDB search
    try:
        # Try without filter first
        results = await client.chroma_search(
            query="sailing",
            k=CONFIG.corpus_chroma_search_k,
            filter=None
        )
        
        logger.info(f"Number of results (no filter): {len(results)}")
        
        # Try with filter
        results_with_filter = await client.chroma_search(
            query="sailing",
            k=CONFIG.corpus_chroma_search_k,
            filter={"topics": "General Sailing Forum"}
        )
        
        logger.info(f"Number of results (with filter): {len(results_with_filter)}")
        
        # Log the metadata fields in the first result if available
        if results:
            logger.info(f"First result metadata keys: {results[0].metadata.keys()}")
            if "topics" in results[0].metadata:
                logger.info(f"Topics field value: {results[0].metadata['topics']}")
            elif "topic" in results[0].metadata:
                logger.info(f"Topic field value: {results[0].metadata['topic']}")
            else:
                logger.warning("No topic/topics field in metadata")
        
    except Exception as e:
        logger.error(f"Error searching ChromaDB: {e}")
        raise

# Check the ChromaDB collection directly
def test_chroma_collection_info():
    """Get information about the ChromaDB collection directly."""
    # Get the services URL from environment or use default
    services_url = os.getenv("SERVICES_URL", "http://localhost:8081")
    
    # Make a request to the health endpoint to get collection info
    try:
        response = requests.get(f"{services_url}/health")
        
        # Log the response
        logger.info(f"Health response: {response.json()}")
        
        # Check if we got collection info
        data = response.json()
        if "chroma_documents" in data:
            logger.info(f"ChromaDB document count: {data['chroma_documents']}")
        else:
            logger.warning("No ChromaDB document count in health response")
            
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise

if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__]) 