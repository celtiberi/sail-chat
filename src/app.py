import os
from pathlib import Path
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
from typing import Optional, Tuple, List, Dict, Any, Set, Callable, Literal
import logging
from time import perf_counter
from langchain_core.documents import Document
import numpy as np
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from src.retriever import Retriever
from src.models import State
from langgraph.graph import START, StateGraph
from src.visual_index.search import VisualSearch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import atexit
import platform
from src.session_manager import SessionManager
from src.visual_index.index_provider import IndexProvider
from utils.logger import ConversationLogger
import uuid

# Check if running on Apple Silicon and configure PyTorch appropriately
is_apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'
if is_apple_silicon:
    try:
        import torch
        print(f"Running on Apple Silicon: {is_apple_silicon}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch has MPS: {hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available')}")
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available'):
            print(f"MPS is available: {torch.mps.is_available()}")
            print("Note: Using CPU device for compatibility with memory-mapped tensor loading")
    except ImportError:
        print("PyTorch not available")

# Import centralized configuration
from src.config import (
    APP_CONFIG as CONFIG,
    RETRIEVER_CONFIG,
    VISUAL_CONFIG,
    UserSettings,
    WORKSPACE_ROOT
)

# Configure logging
LOG_SEPARATOR = "=" * 50
log_level = os.getenv("LOG_LEVEL", "INFO")
print(f"Setting up logging with level: {log_level}")
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("forum_chat.log")
    ],
    force=True  # Override any existing logging configuration
)
# Set the root logger level as well
logging.getLogger().setLevel(log_level)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Print a test message to verify logging is working
logger.debug("DEBUG logging is enabled")
logger.info("INFO logging is enabled")

# Initialize shared resources at application startup
# This ensures we only load resource-intensive components once
try:
    logger.info("Initializing shared resources")
    
    # Initialize VisualSearch shared index
    logger.info("Initializing shared VisualSearch index - START")
    logger.info(f"Visual index path: {VISUAL_CONFIG.INDEX_ROOT}")
    logger.info(f"Visual model name: {VISUAL_CONFIG.MODEL_NAME}")
    logger.info(f"Visual device: {VISUAL_CONFIG.DEVICE}")
    
    # Check if the index is already loaded (for hot reloading scenarios)
    if IndexProvider._index_instance is None:
        logger.info("Index not found in memory, initializing...")
        start_time = perf_counter()
        VisualSearch.initialize_shared_index()
        end_time = perf_counter()
        logger.info(f"VisualSearch shared index initialized successfully in {end_time - start_time:.2f} seconds")
    else:
        logger.info("VisualSearch shared index already loaded, skipping initialization")
    
    logger.info(f"Memory usage after VisualSearch initialization: {os.popen('ps -o rss -p %d | tail -n1' % os.getpid()).read().strip()} KB")
    
    # Initialize ChromaDB - this is the only place ChromaDB is initialized in the application
    logger.info("Initializing shared ChromaDB instance - START")
    logger.info(f"ChromaDB path: {RETRIEVER_CONFIG.get_db_path()}")
    logger.info(f"ChromaDB collection: {RETRIEVER_CONFIG.forum_collection}")
    
    start_time = perf_counter()
    chroma_path = RETRIEVER_CONFIG.get_db_path()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Embeddings model loaded")
    
    SHARED_CHROMA = Chroma(
        persist_directory=chroma_path,
        collection_name=RETRIEVER_CONFIG.forum_collection,
        embedding_function=embeddings
    )
    end_time = perf_counter()
    
    logger.info(f"ChromaDB initialized successfully at {chroma_path} in {end_time - start_time:.2f} seconds")
    logger.info(f"Memory usage after ChromaDB initialization: {os.popen('ps -o rss -p %d | tail -n1' % os.getpid()).read().strip()} KB")
    logger.info("All shared resources initialized successfully")
    
except Exception as e:
    logger.error(f"Error initializing shared resources: {str(e)}", exc_info=True)
    SHARED_CHROMA = None

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def create_chain(retriever: Retriever):
    """Create a LangGraph chain with validation."""
    # Create StateGraph with sequential execution
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("analyze", retriever.analyze_query)
    graph_builder.add_node("retrieve", retriever.retrieve)
    graph_builder.add_node("generate", retriever.generate)
    graph_builder.add_node("reject", retriever.reject_query)
    
    # Add edges with a clear sequential flow
    graph_builder.add_edge(START, "analyze")
    
    
    
    # Continue the sequential flow
    graph_builder.add_edge("analyze", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    
    # Set reject and generate as finish points
    graph_builder.set_finish_point("reject")
    graph_builder.set_finish_point("generate")
    
    return graph_builder.compile()

@cl.on_chat_start
async def on_chat_start():
    try:
        # Initialize session ID when chat starts
        cl.user_session.set("session_id", str(uuid.uuid4()))
        
        # Initialize the session manager
        session = SessionManager(State)
        cl.user_session.set("session", session)
        
        # Initialize LLM with streaming enabled
        llm = ChatGoogleGenerativeAI(
            model=CONFIG.llm.model_name,
            temperature=CONFIG.llm.temperature,
            max_output_tokens=CONFIG.llm.max_tokens,
            top_p=CONFIG.llm.top_p,
            streaming=True,
        )
        
        # Create a new VisualSearch instance for this user session
        visual_search = VisualSearch()
        
        # Create retriever with shared resources
        retriever = Retriever(
            llm=llm,
            visual_search=visual_search,
            forum_store=SHARED_CHROMA
        )
        
        # Create chain with State class as schema
        chain = create_chain(retriever)
        
        # Store chain and empty chat history
        cl.user_session.set("chain", chain)
        cl.user_session.set("state", State(question=""))
        cl.user_session.set("settings", UserSettings())
        cl.user_session.set("visual_search", visual_search)
        
        await cl.Message(
            content="Ready to answer questions about sailing and boating!",
            author="Assistant"
        ).send()
        
    except Exception as e:
        await handle_error(e, "chat initialization")

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
        

    return [
        cl.Text(
            name=f"Source {idx}",
            content=format_content(doc),
            display="inline",
            language="markdown"
        )
        for idx, doc in enumerate(docs, 1)
    ]

async def create_detailed_step(name: str, description: str = None, show_input: bool = True) -> cl.Step:
    """Create a detailed step with proper configuration for visualization.
    
    Args:
        name: Name of the step
        description: Optional description of what the step does
        show_input: Whether to show the input in the UI
        
    Returns:
        A configured Chainlit Step
    """
    # Create step with proper type for visualization
    step = cl.Step(
        name=name,
        type="tool" if show_input else "run",
        show_input=show_input
    )
    
    # Add description if provided
    if description:
        step.output = description
    
    # Send the step to display it in the UI
    await step.send()
    
    return step

@cl.on_message
async def on_message(message: cl.Message):
    try:
        # Get session ID from user session
        session_id = cl.user_session.get("session_id")
        
        # Process the message and get response
        response = await main(message)  # Use the existing main function
        
        # Log the interaction
        conversation_logger.log_interaction(
            session_id=session_id,
            user_message=message.content,
            assistant_message=response.content if hasattr(response, 'content') else str(response),
            metadata={
                "message_id": message.id,
                "timestamp": message.timestamp,
                "has_attachments": bool(message.attachments),
                "attachments": [a.name for a in message.attachments] if message.attachments else []
            }
        )
        
        return response
    except Exception as e:
        await handle_error(e, "message processing")

async def main(message: cl.Message):
    try:
        # Get session from user session
        session = cl.user_session.get("session")
        if not session:
            raise ValueError("Session not initialized")
            
        chain = cl.user_session.get("chain")
        if not chain:
            raise ValueError("Chain not initialized")
        
        # Create state for this retrieval process
        session.model.question = message.content
        
        # Initialize empty dictionaries for steps and tracking
        steps = {}
        updated_steps = {"analyze": False, "retrieve": False, "generate": False, "reject": False}
        
        # Store the tracking dictionary in the session
        session.model.updated_steps = updated_steps
        
        # Store the steps dictionary in the session for the nodes to access
        session.model.steps = steps
        
        # Create a message object for streaming that we'll use internally
        temp_msg = cl.Message(content="", author="Assistant")
        await temp_msg.send()  # Send the empty message so we can stream to it
        session.model.current_message = temp_msg  # Store for streaming in the generate step
        
        # Run the chain
        result = await chain.ainvoke(session.model.model_dump())
        
        # Batch updates
        with session.batch_update() as state:
            state.running_context = result["running_context"]
            state.chat_history = result["chat_history"]
            state.chat_history.append({"role": "human", "content": message.content})
            state.chat_history.append({"role": "assistant", "content": result["answer"]})
            state.current_message = None
            state.steps = {}
            state.updated_steps = {}
        
        # Add sources if needed
        settings = cl.user_session.get("settings", UserSettings())
        if settings.show_sources and result.get("current_context"):
            temp_msg.elements = format_docs_as_sources(result["current_context"])
            await temp_msg.update()
            
        return temp_msg
        
    except Exception as e:
        logger.error("Error in message processing", exc_info=True)
        await handle_error(e, "processing")

@cl.on_chat_end
async def end():
    try:
        # Clean up user-specific resources
        visual_search = cl.user_session.get("visual_search")
        if visual_search:
            logger.info("Cleaning up user's VisualSearch instance")
            visual_search.close()
            logger.info("User's VisualSearch instance cleaned up")
        
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
        if "debug_mode" in settings:
            current.debug_mode = bool(settings["debug_mode"])
            
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
    settings = cl.user_session.get("settings", UserSettings())
    if settings and settings.debug_mode:
        user_msg = f"{user_msg}\n\nDebug info: {str(error)}"
    
    await cl.Message(
        content=user_msg,
        author="System",
        type="error"
    ).send()

# Register cleanup function to ensure resources are released on application exit
def cleanup_resources():
    try:
        logger.info("Application shutting down, cleaning up resources...")
        
        # Close the shared index
        from visual_index.search import VisualSearch
        logger.info("Closing shared VisualSearch index")
        VisualSearch.close_shared_index()
        
        # Clean up ChromaDB
        if 'SHARED_CHROMA' in globals():
            try:
                logger.info("Closing ChromaDB...")
                # ChromaDB handles cleanup internally when Python exits
                # but we can explicitly log it
                logger.info("ChromaDB will be closed during Python shutdown")
            except Exception as e:
                logger.error(f"Error with ChromaDB cleanup: {str(e)}")
        
        logger.info("All resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during application cleanup: {str(e)}", exc_info=True)

# Register the cleanup function to run when the application exits
atexit.register(cleanup_resources)

# Add this at the end of the file
if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    
    # Run the Chainlit application
    run_chainlit(__file__)

# Initialize the conversation logger
conversation_logger = ConversationLogger()
