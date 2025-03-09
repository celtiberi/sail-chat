"""
Integration test for ChromaDB initialization.

This test checks if ChromaDB is properly initialized with documents.
"""

import os
import sys
import pytest
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the configuration
from src.config import CONFIG

@pytest.fixture
def chroma_db():
    """Create a ChromaDB instance for testing."""
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG.embedding_model)
    
    # Create or load the ChromaDB instance
    db = Chroma(
        collection_name="forum_content",
        embedding_function=embeddings,
        persist_directory=str(CONFIG.chroma_db_dir)
    )
    
    return db

def test_chroma_has_documents(chroma_db):
    """Test that ChromaDB has documents."""
    # Get the number of documents in the collection
    count = chroma_db._collection.count()
    
    # Log the count for debugging
    print(f"ChromaDB collection has {count} documents")
    
    # Assert that there are documents in the collection
    assert count > 0, "ChromaDB collection is empty"
    
    # Get a sample of documents to check their structure
    if count > 0:
        # Get some documents
        docs = chroma_db.similarity_search("test", k=min(5, count))
        
        # Log the documents for debugging
        print(f"Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"Document {i+1}:")
            print(f"  Content: {doc.page_content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
        
        # Assert that the documents have the expected structure
        for doc in docs:
            assert hasattr(doc, 'page_content'), "Document missing page_content"
            assert hasattr(doc, 'metadata'), "Document missing metadata" 