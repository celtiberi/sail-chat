#!/usr/bin/env python
"""
Run the Corpus Service

This script runs the Corpus Service which provides access to:
- Visual search index
- ChromaDB for text embeddings
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to Python path")

if __name__ == "__main__":
    try:
        # Get the port from the environment or use a default
        port = int(os.getenv("SERVICES_PORT", 8081))
        
        # Import uvicorn here after path setup
        import uvicorn
        
        # Run the service
        logger.info(f"Starting Corpus Service on port {port}")
        logger.info(f"Current directory: {os.getcwd()}")
        
        uvicorn.run("services.corpus_service.service:app", 
                    host="0.0.0.0", 
                    port=port,
                    log_level="info")
    except Exception as e:
        logger.error(f"Error starting Corpus Service: {e}", exc_info=True)
        sys.exit(1) 