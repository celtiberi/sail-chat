#!/usr/bin/env python
"""
Test script for the Combined Services.

This script tests the various endpoints of the Combined Services
to ensure they are working correctly.
"""

import os
import sys
import asyncio
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the service client
from services.client import ServiceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_health_check(client):
    """Test the health check endpoint."""
    logger.info("Testing health check endpoint...")
    try:
        is_healthy = await client.health_check()
        if is_healthy:
            logger.info("✅ Health check passed")
        else:
            logger.error("❌ Health check failed")
    except Exception as e:
        logger.error(f"❌ Error during health check: {e}")

async def test_visual_search(client):
    """Test the visual search endpoint."""
    logger.info("Testing visual search endpoint...")
    try:
        # Test with a simple query
        query = "sailing boat"
        results, paths = await client.visual_search(query=query, k=3)
        
        logger.info(f"Visual search returned {len(results)} results")
        if results:
            logger.info("✅ Visual search test passed")
            # Print the first result
            result = results[0]
            logger.info(f"First result: doc_id={result.doc_id}, file_name={result.file_name}, score={result.score}")
            logger.info(f"Metadata: {json.dumps(result.metadata, default=str)}")
        else:
            logger.warning("⚠️ Visual search returned no results")
    except Exception as e:
        logger.error(f"❌ Error during visual search test: {e}")

async def test_chroma_search(client):
    """Test the ChromaDB search endpoint."""
    logger.info("Testing ChromaDB search endpoint...")
    try:
        # Test with a simple query
        query = "sailing"
        docs = await client.chroma_search(query=query, k=5)
        
        logger.info(f"ChromaDB search returned {len(docs)} documents")
        if docs:
            logger.info("✅ ChromaDB search test passed")
            # Print the first document
            doc = docs[0]
            logger.info(f"First document: {doc.page_content[:100]}...")
            logger.info(f"Metadata: {json.dumps(doc.metadata, default=str)}")
        else:
            logger.warning("⚠️ ChromaDB search returned no documents")
            
            # Try a different query
            query = "boat"
            logger.info(f"Trying a different query: {query}")
            docs = await client.chroma_search(query=query, k=5)
            
            logger.info(f"ChromaDB search returned {len(docs)} documents")
            if docs:
                logger.info("✅ ChromaDB search test passed with alternative query")
                # Print the first document
                doc = docs[0]
                logger.info(f"First document: {doc.page_content[:100]}...")
                logger.info(f"Metadata: {json.dumps(doc.metadata, default=str)}")
            else:
                logger.error("❌ ChromaDB search returned no documents for any query")
                
                # Check if the ChromaDB collection is empty
                logger.info("Checking if the ChromaDB collection is empty...")
                # This would require a direct check of the ChromaDB collection
                # For now, we'll just log a message
                logger.info("Please check if the ChromaDB collection has been properly initialized with documents")
    except Exception as e:
        logger.error(f"❌ Error during ChromaDB search test: {e}")

async def main():
    """Run the tests."""
    # Create a client
    client = ServiceClient()
    
    try:
        # Test health check
        await test_health_check(client)
        
        # Test visual search
        await test_visual_search(client)
        
        # Test ChromaDB search
        await test_chroma_search(client)
    finally:
        # Close the client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 