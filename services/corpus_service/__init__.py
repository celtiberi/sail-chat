"""
Corpus Service Package

This package provides a service for accessing the knowledge corpus including:
- Visual search index
- ChromaDB for text embeddings

It includes both the service implementation and a client for interacting with the service.
"""

from .client import CorpusClient, SearchResult 