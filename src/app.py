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
from retriever import Retriever
from models import State
from langgraph.graph import START, StateGraph
from visual_index.search import VisualSearch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import atexit

from dataclasses import asdict
from session_manager import SessionManager

# Import centralized configuration
from config import (
    APP_CONFIG as CONFIG,
    RETRIEVER_CONFIG,
    VISUAL_CONFIG,
    UserSettings,
    WORKSPACE_ROOT
)

# Configure logging
LOG_SEPARATOR = "=" * 50
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("forum_chat.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize shared resources at application startup
# This ensures we only load resource-intensive components once
try:
    logger.info("Initializing shared resources")
    
    # Initialize VisualSearch shared index
    logger.info("Initializing shared VisualSearch index")
    VisualSearch.initialize_shared_index()
    logger.info("VisualSearch shared index initialized successfully")
    
    # Initialize ChromaDB - this is the only place ChromaDB is initialized in the application
    logger.info("Initializing shared ChromaDB instance")
    chroma_path = RETRIEVER_CONFIG.get_db_path()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    SHARED_CHROMA = Chroma(
        persist_directory=chroma_path,
        collection_name=RETRIEVER_CONFIG.forum_collection,
        embedding_function=embeddings
    )
    logger.info(f"ChromaDB initialized successfully at {chroma_path}")
    
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

def should_continue(state: State) -> Literal["retrieve", "reject"]:
    """Determine next node based on validation result."""
    return "retrieve" if state.is_sailing_related else "reject"

def create_chain(retriever: Retriever):
    """Create a LangGraph chain with validation."""
    # Create StateGraph with sequential execution
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("validate", retriever.validate_query)
    graph_builder.add_node("analyze", retriever.analyze_query)
    graph_builder.add_node("retrieve", retriever.retrieve)
    graph_builder.add_node("generate", retriever.generate)
    graph_builder.add_node("reject", retriever.reject_query)
    
    # Add edges with a clear sequential flow
    graph_builder.add_edge(START, "validate")
    
    # Define conditional paths after validation
    graph_builder.add_conditional_edges(
        "validate",
        should_continue,
        {
            "retrieve": "analyze",  # If should_continue returns "retrieve", go to analyze
            "reject": "reject"      # If should_continue returns "reject", go to reject
        }
    )
    
    # Continue the sequential flow
    graph_builder.add_edge("analyze", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    
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
        
        # Create a new VisualSearch instance for this user session
        # This uses the shared index but has its own instance for thread safety
        visual_search = VisualSearch()
        logger.info("VisualSearch instance created for this session")
        
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
        cl.user_session.set("visual_search", visual_search)  # Store for cleanup later

        # Add a comment to explain the use of RETRIEVER_CONFIG
        # Using retriever configuration imported from retriever.py
        logger.info(f"Using forum collection: {RETRIEVER_CONFIG.forum_collection}")
        logger.info(f"Retrieving up to {RETRIEVER_CONFIG.num_forum_results} forum documents")
        
        # Update the message to reflect the configuration source
        await cl.Message(
            content="Ready to answer questions about sailing and boating!",
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
        settings = cl.user_session.get("settings", UserSettings())
        if settings.show_sources and result.get("current_context"):
            msg.elements = format_docs_as_sources(result["current_context"])
        await msg.update()
        
        
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
    if settings.debug_mode:
        user_msg = f"{user_msg}\n\nDebug info: {str(error)}"
    
    await cl.Message(
        content=user_msg,
        author="System",
        type="error"
    ).send()

# Add this at the end of the file
if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    
    # Register shutdown handler to clean up resources
    def cleanup_resources():
        """Clean up shared resources when the application exits."""
        logger.info("Cleaning up shared resources...")
        
        # Clean up VisualSearch shared index
        try:
            logger.info("Closing VisualSearch shared index...")
            VisualSearch.close_shared_index()
            logger.info("VisualSearch shared index closed successfully")
        except Exception as e:
            logger.error(f"Error closing VisualSearch shared index: {str(e)}")
        
        # Clean up ChromaDB
        if SHARED_CHROMA:
            try:
                logger.info("Closing ChromaDB...")
                # ChromaDB handles cleanup internally when Python exits
                # but we can explicitly log it
                logger.info("ChromaDB will be closed during Python shutdown")
            except Exception as e:
                logger.error(f"Error with ChromaDB cleanup: {str(e)}")
    
    # Register the cleanup function to run on exit
    atexit.register(cleanup_resources)
    
    # Run the Chainlit application
    run_chainlit(__file__)
