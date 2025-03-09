#!/usr/bin/env python
"""
Script to start the Combined Service.

This script starts the Combined Service as a standalone process.
It loads both the visual index and ChromaDB once and keeps them in memory,
serving requests through a REST API.

Usage:
    python scripts/start_services.py [--port PORT]
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """Start the Combined Service."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start the Combined Service")
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVICES_PORT", 8081)),
                        help="Port to run the service on")
    parser.add_argument("--host", type=str, default=os.getenv("SERVICES_HOST", "0.0.0.0"),
                        help="Host to run the service on")
    args = parser.parse_args()
    
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Log the startup
    logger.info(f"Starting Combined Service on {args.host}:{args.port}")
    
    # Import and run the service
    import uvicorn
    uvicorn.run("services.service:app", host=args.host, port=args.port)

if __name__ == "__main__":
    main() 