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
from models import ForumTopic, State
from session_manager import SessionManager
from langgraph.graph import START, StateGraph
import platform
from utils.logger import ConversationLogger
import uuid
from langchain.schema import SystemMessage, HumanMessage

# Import centralized configuration
from src.config import (
    APP_CONFIG,
    CONFIG,
    RETRIEVER_CONFIG,
    VISUAL_CONFIG,
    UserSettings,
)

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("forum_chat.log")
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Print a test message to verify logging is working
logger.debug("DEBUG logging is enabled")
logger.info("INFO logging is enabled")

# Initialize conversation logger
conversation_logger = ConversationLogger()

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
        
        # Initialize the session manager with the State class
        session = SessionManager(State)
        cl.user_session.set("session", session)
        
        # Create retriever (LLM will be created internally)
        retriever = Retriever()
        
        # Create chain with State class as schema
        chain = create_chain(retriever)
        
        # Store chain and empty chat history
        cl.user_session.set("chain", chain)
        cl.user_session.set("state", State(question=""))
        cl.user_session.set("settings", UserSettings())
        
        # Send a welcome message
        await cl.Message(
            content="Ready to answer questions about sailing and boating!",
            author="Sailors Parrot"
        ).send()
        
    except Exception as e:
        logger.error(f"Error during chat initialization: {e}")
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
        
        # Debug logging to see what attributes are available
        logger.debug(f"Message attributes: {dir(message)}")
        
        # Process the message and get response
        response = await main(message)  # Use the existing main function
        
        # Create metadata with only the most basic attributes we know exist
        metadata = {
            "message_id": getattr(message, 'id', None),  # Safely get id if it exists
        }
        
        # Log the interaction
        conversation_logger.log_interaction(
            session_id=session_id,
            user_message=message.content,
            assistant_message=response.content if hasattr(response, 'content') else str(response),
            metadata=metadata
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
        temp_msg = cl.Message(content="", author="Sailors Parrot")
        await temp_msg.send()  # Send the empty message so we can stream to it
        session.model.current_message = temp_msg  # Store for streaming in the generate step
        
        # Run the chain
        result = await chain.ainvoke(session.model.model_dump())
        
        # Batch updates
        with session.batch_update() as state:
            state.chat_history = result["chat_history"]
            state.chat_history.append({"role": "human", "content": message.content})
            state.chat_history.append({"role": "assistant", "content": result["answer"]})
            state.current_message = None
            state.steps = {}
            state.updated_steps = {}
        
        # Add sources if needed
        # settings = cl.user_session.get("settings", UserSettings())
        # if settings.show_sources and result.get("current_context"):
        #     temp_msg.elements = format_docs_as_sources(result["current_context"])
        #     await temp_msg.update()
            
        return temp_msg
        
    except Exception as e:
        logger.error("Error in message processing", exc_info=True)
        await handle_error(e, "processing")

@cl.on_chat_end
async def end():
    try:
        # Get the retriever from the user session
        chain = cl.user_session.get("chain")
        if chain and hasattr(chain, "retriever"):
            retriever = chain.retriever
            
            # Close the service client if it exists
            if hasattr(retriever, "service_client"):
                logger.info("Closing service client connection")
                await retriever.service_client.close()
                logger.info("Service client connection closed")
        
        await cl.Message(
            content="Chat session ended. Thank you for using the assistant!",
            author="Sailors Parrot"
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

# No need for direct Chainlit run since it's mounted by FastAPI

