"""
Pytest configuration for integration tests.
"""

import os
import sys
import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the service client
from services.client import ServiceClient

# Fixture for the service client
@pytest_asyncio.fixture
async def client():
    """Create a service client for testing."""
    client = ServiceClient()
    try:
        yield client
    finally:
        await client.close()

# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest."""
    # Set the default event loop policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy()) 