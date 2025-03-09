#!/usr/bin/env python
"""
Script to directly test ChromaDB functionality.
This script helps diagnose why we're not getting any documents from ChromaDB.
"""

import os
import sys
import json
import logging
from pathlib import Path
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import CONFIG, RETRIEVER_CONFIG

def test_chroma_direct():
    """Test ChromaDB directly using langchain_chroma."""
    logger.info("Testing ChromaDB directly")
    
    # Get the ChromaDB directory
    chroma_dir = CONFIG.chroma_db_dir
    logger.info(f"ChromaDB directory: {chroma_dir}")
    
    # Check if the directory exists
    if not os.path.exists(chroma_dir):
        logger.error(f"ChromaDB directory does not exist: {chroma_dir}")
        return
    
    # Initialize the embedding model
    logger.info(f"Initializing embedding model: {CONFIG.embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG.embedding_model)
    
    # Initialize ChromaDB
    logger.info(f"Initializing ChromaDB with collection: {RETRIEVER_CONFIG.forum_collection}")
    try:
        chroma_db = Chroma(
            collection_name=RETRIEVER_CONFIG.forum_collection,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir)
        )
        
        # Get the collection count
        count = chroma_db._collection.count()
        logger.info(f"ChromaDB collection count: {count}")
        
        if count == 0:
            logger.error("ChromaDB collection is empty!")
            return
        
        # Try a simple search without filter
        logger.info("Performing search without filter")
        results = chroma_db.similarity_search(
            query="sailing",
            k=10
        )
        
        logger.info(f"Search results (no filter): {len(results)}")
        
        # Log the first result if available
        if results:
            logger.info(f"First result content: {results[0].page_content[:100]}...")
            logger.info(f"First result metadata: {results[0].metadata}")
        else:
            logger.warning("No results returned from search without filter")
        
        # Try a search with filter
        logger.info("Performing search with filter")
        
        # First, let's check what metadata fields are available
        logger.info("Checking available metadata fields")
        if results:
            metadata_keys = results[0].metadata.keys()
            logger.info(f"Available metadata fields: {metadata_keys}")
            
            # Check if 'topics' or 'topic' is in the metadata
            if 'topics' in metadata_keys:
                logger.info(f"Topics field value: {results[0].metadata['topics']}")
                
                # Try search with topics filter
                filter_results = chroma_db.similarity_search(
                    query="sailing",
                    k=10,
                    filter={"topics": results[0].metadata['topics']}  # Use the actual value from the first result
                )
                
                logger.info(f"Search results with topics filter: {len(filter_results)}")
            elif 'topic' in metadata_keys:
                logger.info(f"Topic field value: {results[0].metadata['topic']}")
                
                # Try search with topic filter
                filter_results = chroma_db.similarity_search(
                    query="sailing",
                    k=10,
                    filter={"topic": results[0].metadata['topic']}  # Use the actual value from the first result
                )
                
                logger.info(f"Search results with topic filter: {len(filter_results)}")
            else:
                logger.warning("No topic/topics field in metadata")
                
                # Try with a different filter if available
                if metadata_keys:
                    key = list(metadata_keys)[0]
                    value = results[0].metadata[key]
                    logger.info(f"Trying filter with {key}={value}")
                    
                    filter_results = chroma_db.similarity_search(
                        query="sailing",
                        k=10,
                        filter={key: value}
                    )
                    
                    logger.info(f"Search results with {key} filter: {len(filter_results)}")
        
    except Exception as e:
        logger.error(f"Error testing ChromaDB: {e}", exc_info=True)

async def check_service_client():
    """Test the service client's ChromaDB search functionality."""
    logger.info("Testing service client")
    
    # Import the service client
    from services.client import ServiceClient
    
    # Get the services URL from environment or use default
    services_url = os.getenv("SERVICES_URL", "http://localhost:8081")
    logger.info(f"Services URL: {services_url}")
    
    # Create the service client
    client = ServiceClient(services_url)
    
    # Test the health check
    try:
        is_healthy = await client.health_check()
        logger.info(f"Service health: {is_healthy}")
        
        # Test the ChromaDB search
        logger.info("Testing ChromaDB search via service client")
        results = await client.chroma_search(
            query="sailing",
            k=10,
            filter=None
        )
        
        logger.info(f"Number of results (no filter): {len(results)}")
        
        # Log the first result if available
        if results:
            logger.info(f"First result content: {results[0].page_content[:100]}...")
            logger.info(f"First result metadata: {results[0].metadata}")
            
            # Check if 'topics' or 'topic' is in the metadata
            if 'topics' in results[0].metadata:
                topic_value = results[0].metadata['topics']
                logger.info(f"Topics field value: {topic_value}")
                
                # Try with topics filter
                results_with_filter = await client.chroma_search(
                    query="sailing",
                    k=10,
                    filter={"topics": topic_value}
                )
                
                logger.info(f"Number of results (with topics filter): {len(results_with_filter)}")
            elif 'topic' in results[0].metadata:
                topic_value = results[0].metadata['topic']
                logger.info(f"Topic field value: {topic_value}")
                
                # Try with topic filter
                results_with_filter = await client.chroma_search(
                    query="sailing",
                    k=10,
                    filter={"topic": topic_value}
                )
                
                logger.info(f"Number of results (with topic filter): {len(results_with_filter)}")
            else:
                logger.warning("No topic/topics field in metadata")
        else:
            logger.warning("No results returned from service client")
        
    except Exception as e:
        logger.error(f"Error testing service client: {e}", exc_info=True)

async def main():
    """Main async function to run all tests."""
    logger.info("Starting ChromaDB test script")
    
    # Test ChromaDB directly
    test_chroma_direct()
    
    # Test service client
    await check_service_client()
    
    logger.info("ChromaDB test script completed")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 