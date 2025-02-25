import os
from pathlib import Path
from dotenv import load_dotenv
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Set, Callable, Literal
import logging
from time import perf_counter
from langchain_core.documents import Document
import numpy as np
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from retriever import Retriever
from models import State
from langgraph.graph import START, StateGraph

from dataclasses import asdict
from session_manager import SessionManager
from chromadb.config import Settings
from chromadb import HttpClient

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
    forum_collection: str = "forum_content"  # Forum content collection
    book_collection: str = "book_content"    # Book content collection
    num_forum_results: int = 30  # Number of forum documents to retrieve
    num_book_results: int = 20   # Number of book documents to retrieve
    mmr_lambda: float = 0.7      # Balance between relevance and diversity
    fetch_k: int = 20            # Number of candidates for MMR

@dataclass
class ChatConfig:
    """Chat History and Context Configuration"""
    max_history_items: int = 20  # Maximum number of messages to keep
    context_window: int = 5  # Number of recent messages for context
    show_sources: bool = True  # Whether to display source documents
    show_search_stats: bool = True  # Whether to show number of docs found

@dataclass
class VectorDBConfig:
    """Vector Database Configuration"""
    host: str = "localhost"
    port: int = 8000
    collection_location: str = "http://{host}:{port}"  # Template for Chroma URL

    def get_url(self) -> str:
        """Get formatted Chroma URL"""
        return self.collection_location.format(host=self.host, port=self.port)

@dataclass
class AppConfig:
    """Main Application Configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    vectordb: VectorDBConfig = field(default_factory=VectorDBConfig)
    debug_mode: bool = False
    
    def validate(self):
        """Validate configuration values."""
        if self.retriever.num_forum_results < 1:
            raise ValueError("num_forum_results must be positive")
        if self.retriever.num_book_results < 1:
            raise ValueError("num_book_results must be positive")
        if self.chat.max_history_items < 1:
            raise ValueError("max_history_items must be positive")
        if not self.llm.model_name:
            raise ValueError("model_name must be specified")
        if self.llm.temperature < 0 or self.llm.temperature > 1:
            raise ValueError("temperature must be between 0 and 1")
        if not self.vectordb.host:
            raise ValueError("host must be specified for vectordb")

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

def should_continue(state: State) -> Literal["retrieve", "reject"]:
    """Determine next node based on validation result."""
    return "retrieve" if state.is_sailing_related else "reject"

def create_chain(retriever: Retriever):
    """Create a LangGraph chain with validation."""
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("validate", retriever.validate_query)
    graph_builder.add_node("analyze", retriever.analyze_query)
    graph_builder.add_node("retrieve", retriever.retrieve)
    graph_builder.add_node("generate", retriever.generate)
    graph_builder.add_node("reject", retriever.reject_query)
    
    # Add edges
    graph_builder.add_edge(START, "validate")
    
    # Add conditional edge after validation
    graph_builder.add_conditional_edges(
        "validate",
        should_continue
    )
    
    # Add remaining edges
    graph_builder.add_edge("retrieve", "analyze")
    graph_builder.add_edge("analyze", "generate")
    
    # Set reject and generate as finish points
    graph_builder.set_finish_point("reject")
    graph_builder.set_finish_point("generate")
    
    return graph_builder.compile()

@cl.on_chat_start
async def init():
    try:
        logger.info(f"\n{LOG_SEPARATOR}")
        logger.info("Initializing chat session")
        
        session = SessionManager(State)
        cl.user_session.set("session", session)

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
        
        # Create Chroma client for remote server
        chroma_client = HttpClient(
            host=f"http://{CONFIG.vectordb.host}:{CONFIG.vectordb.port}",
            ssl=False
        )
        logger.info(f"Connected to Chroma at {CONFIG.vectordb.get_url()}")
        
        # Initialize vector stores
        forum_db = Chroma(
            client=chroma_client,
            collection_name=CONFIG.retriever.forum_collection,
            embedding_function=embeddings
        )

        
        logger.info("Vector stores initialized")
        
        # Create retriever with both collections
        retriever = Retriever(
            forum_store=forum_db,
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

def format_docs_as_sources(docs: List[Document]) -> List[cl.Text]:
    """Format documents as source elements."""
    def format_content(doc: Document) -> str:
        """Format content based on document type."""
        if doc.metadata.get('thread_id'):  # Forum post
            return (
                f"**Forum Post**\n"
                f"Topic: {doc.metadata.get('topic', 'General Discussion')}\n"
                f"Thread: {doc.metadata.get('thread_id')}\n"
                f"Forum: {doc.metadata.get('forum_id', 'Unknown')}\n\n"
                f"{doc.page_content}"
            )
        else:  # Book excerpt
            return (
                f"**Book: {doc.metadata.get('book_title', 'Sailing Reference')}**\n"
                f"Chapter: {doc.metadata.get('chapter', 'N/A')}\n"
                f"Section: {doc.metadata.get('section', 'N/A')}\n"
                f"Page: {doc.metadata.get('page', 'N/A')}\n\n"
                f"{doc.page_content}"
            )

    return [
        cl.Text(
            name=f"Source {idx}",
            content=format_content(doc),
            display="inline",
            language="markdown"
        )
        for idx, doc in enumerate(docs, 1)
    ]

@cl.on_message
async def main(message: cl.Message):
    try:
        session = cl.user_session.get("session")
        chain = cl.user_session.get("chain")

        
        # Create response message
        msg = cl.Message(content="", author="Assistant")
        await msg.send()
        
        # Create state for this retrieval process
        session.model.question = message.content
        result = await chain.ainvoke(session.model.model_dump())
        
        # Batch updates
        with session.batch_update() as state:
            state.running_context = result["running_context"]
            state.chat_history = result["chat_history"]
            state.chat_history.append({"role": "human", "content": message.content})
            state.chat_history.append({"role": "assistant", "content": result["answer"]})
        
        # Update response
        msg.content = result["answer"]
        if CONFIG.chat.show_sources and result.get("current_context"):
            msg.elements = format_docs_as_sources(result["current_context"])
        await msg.update()
        
        
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

# Add this at the end of the file
if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
