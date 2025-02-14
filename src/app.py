import os
from pathlib import Path
from dotenv import load_dotenv
import chainlit as cl
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import sys
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Set, Callable
import logging
from time import perf_counter
from langchain_core.documents import Document
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from langchain.retrievers import (
    MultiQueryRetriever,
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.storage import InMemoryStore
from langchain_core.retrievers import BaseRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.schema import format_document
from langchain.schema.messages import get_buffer_string
from langchain.schema.runnable import RunnableParallel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field
from functools import lru_cache
from contextlib import asynccontextmanager
from retriever import Retriever, State
from langgraph.graph import START, StateGraph

# ======================
# Application Configuration
# ======================

@dataclass
class LLMConfig:
    """Language Model Configuration"""
    model_name: str = "gemini-2.0-flash"  # Model to use
    temperature: float = 0.7  # Higher = more creative, Lower = more focused
    top_p: float = 0.8  # Nucleus sampling parameter
    max_tokens: Optional[int] = None  # Max output length (None = model default)

@dataclass
class RetrieverConfig:
    """Document Retrieval Configuration"""
    content_collection: str = "forum_content"  # Main content collection
    hierarchy_collection: str = "forum_hierarchies"  # Hierarchy collection
    num_hierarchy_results: int = 3  # Number of hierarchy documents to retrieve
    num_content_results: int = 50  # Number of content documents to retrieve
    mmr_lambda: float = 0.7  # Balance between relevance and diversity
    fetch_k: int = 20        # Number of candidates for MMR

@dataclass
class ChatConfig:
    """Chat History and Context Configuration"""
    max_history_items: int = 20  # Maximum number of messages to keep
    context_window: int = 5  # Number of recent messages for context
    show_sources: bool = False  # Whether to display source documents
    show_search_stats: bool = True  # Whether to show number of docs found

@dataclass
class AppConfig:
    """Main Application Configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    debug_mode: bool = False
    
    def validate(self):
        """Validate configuration values."""
        if self.retriever.num_hierarchy_results < 1:
            raise ValueError("num_hierarchy_results must be positive")
        if self.retriever.num_content_results < 1:
            raise ValueError("num_content_results must be positive")
        if self.chat.max_history_items < 1:
            raise ValueError("max_history_items must be positive")
        if not self.llm.model_name:
            raise ValueError("model_name must be specified")
        if self.llm.temperature < 0 or self.llm.temperature > 1:
            raise ValueError("temperature must be between 0 and 1")

@dataclass
class UserSettings:
    show_sources: bool = True
    show_stats: bool = True

# Create configuration instance
CONFIG = AppConfig()

# Load environment variables
if not load_dotenv():
    print("Warning: No .env file found", file=sys.stderr)

# Validate required environment variables
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Please set it in your .env file"
    )

# Get vectordb path relative to current file
VECTORDB_PATH = Path(__file__).parent.parent.parent / "forum-scrape" / "vectordb"

# Add these constants at the top
PREVIEW_LENGTH: int = 200  # Length of content previews in logs
LOG_SEPARATOR: str = "=" * 50  # Separator for log sections

# Add at top of file after other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('forum_chat.log')
    ]
)
logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

class ListRetriever(BaseRetriever):
    """Simple retriever that returns documents from a list."""
    
    documents: List[Document] = Field(default_factory=list)
    
    def __init__(self, documents: List[Document]):
        super().__init__(documents=documents)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.documents
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents

def create_chain(retriever: Retriever):
    """Create a LangGraph chain for retrieval and generation."""
    # Create the graph with State class as schema
    graph_builder = StateGraph(State)  # Pass the class, not instance
    
    # Add nodes for retrieval and generation
    graph_builder.add_node("analyze", retriever.analyze_query)
    graph_builder.add_node("retrieve", retriever.retrieve)
    graph_builder.add_node("generate", retriever.generate)
    
    # Add edges
    graph_builder.add_edge(START, "analyze")
    graph_builder.add_edge("analyze", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    
    # Compile the graph
    return graph_builder.compile()

@cl.on_chat_start
async def init():
    try:
        logger.info(f"\n{LOG_SEPARATOR}")
        logger.info("Initializing chat session")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=CONFIG.llm.model_name,
            temperature=CONFIG.llm.temperature,
            max_output_tokens=CONFIG.llm.max_tokens,
            top_p=CONFIG.llm.top_p,
        )
        logger.info("LLM initialized")
        
        # Initialize embeddings for LangChain
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB client
        client = PersistentClient(path=str(VECTORDB_PATH))
        
        # Get collections using LangChain embeddings
        hierarchy_db = Chroma(
            client=client,
            collection_name=CONFIG.retriever.hierarchy_collection,
            embedding_function=embeddings  # Use LangChain embeddings
        )
        
        content_db = Chroma(
            client=client,
            collection_name=CONFIG.retriever.content_collection,
            embedding_function=embeddings  # Use LangChain embeddings
        )
        logger.info("Vector stores initialized")
        
        # Create retriever
        retriever = Retriever(
            vector_store=content_db,
            llm=llm,
        )
        
        # Create chain with State class as schema
        chain = create_chain(retriever)
        
        # Store chain and empty chat history
        cl.user_session.set("chain", chain)
        cl.user_session.set("state", State(question=""))

        await cl.Message(
            content="Ready to answer questions about the forum data!",
            author="Assistant"
        ).send()
        
    except Exception as e:
        await handle_error(e, "initialization")

@cl.on_message
async def main(message: cl.Message):
    try:
        start_time = perf_counter()
        logger.info(f"\n{LOG_SEPARATOR}")
        logger.info(f"Processing message: {message.content}")
        
        # Get chain
        chain = cl.user_session.get("chain")
        if not chain:
            logger.error("Chain not found in session")
            raise RuntimeError("Session not properly initialized")
        
        # Create response message
        logger.info("Sending initial empty message")
        msg = cl.Message(content="", author="Assistant")
        await msg.send()
        
        # Create state for this retrieval process
        state = cl.user_session.get("state")
        state.question = message.content
        result = await chain.ainvoke(state.model_dump())
        
        # Update state with result
        state.running_context = result["running_context"]
        state.chat_history = result["chat_history"]

        state.chat_history.append({"role": "human", "content": message.content})
        state.chat_history.append({"role": "assistant", "content": result["answer"]})
        
        cl.user_session.set("state", state)
        
        # Update response
        logger.info("Updating response message")
        msg.content = result["answer"]
        if CONFIG.chat.show_sources:
            msg.elements = prepare_source_elements(result.get("context", []))
        await msg.update()
        
        logger.info(f"Message processing completed in {perf_counter() - start_time:.2f}s")
        
    except Exception as e:
        logger.error("Error in message processing", exc_info=True)
        await handle_error(e, "processing")

@cl.on_chat_end
async def end():
    try:
        # ChromaDB handles cleanup internally
        await cl.Message(
            content="Chat session ended. Thank you for using the assistant!",
            author="Assistant"
        ).send()
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@cl.on_stop
async def stop():
    await cl.Message(
        content="Processing stopped. Feel free to ask another question!",
        author="Assistant"
    ).send()

# Add new utility endpoints
@cl.on_settings_update
async def setup_settings(settings: dict):
    """Update user settings with validation."""
    try:
        current = cl.user_session.get("settings", UserSettings())
        
        # Validate and update settings
        if "show_sources" in settings:
            current.show_sources = bool(settings["show_sources"])
        if "show_stats" in settings:
            current.show_stats = bool(settings["show_stats"])
            
        cl.user_session.set("settings", current)
        
        await cl.Message(
            content="Settings updated successfully",
            author="System",
            type="info"
        ).send()
    except Exception as e:
        await handle_error(e, "settings update")

@cl.on_chat_resume
async def resume_chat():
    """Handle chat resume"""
    await cl.Message(
        content="Restored previous conversation state.",
        author="System",
        type="info"
    ).send()

def prepare_source_elements(docs: List[Document]) -> List[cl.Text]:
    """Create source elements for display."""
    return [
        cl.Text(
            name=f"Source {idx}",
            content=f"**Thread {doc.metadata.get('thread_id')} from Forum {doc.metadata.get('forum_id')}**\n"
                   f"Topic: {doc.metadata.get('topic', 'Unknown Topic')}\n\n"
                   f"{doc.page_content}",
            display="inline",
            language="markdown"
        )
        for idx, doc in enumerate(docs, 1)
    ]

async def handle_error(error: Exception, context: str):
    """Handle errors consistently."""
    error_msg = f"Error during {context}: {str(error)}"
    logger.error(error_msg, exc_info=True)
    
    user_msg = "An error occurred while processing your request."
    if CONFIG.debug_mode:
        user_msg = f"{user_msg}\n\nDebug info: {str(error)}"
    
    await cl.Message(
        content=user_msg,
        author="System",
        type="error"
    ).send()
