import os
import asyncio
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
from tools import wind_data_tool, documents_tool
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages.tool import ToolMessage

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

@tool
async def process_wind_data(
    min_lat: float = 37.5, 
    max_lat: float = 42.5, 
    min_lon: float = -72.5, 
    max_lon: float = -67.5) -> None:
    """Get current wind data and visualization for a specific geographical region.
    Use this tool when the user is asking about current wind conditions, weather, or sailing conditions in a specific area."""
    
    # Call the wind_data_tool with the arguments as a dictionary
    wind_data = await wind_data_tool({
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon
    })
    
    elements = []
    output = ""
    if "error" in wind_data:
        output = wind_data['error']
    else:
       elements.append(
            cl.Image(
                name="wind_map",
                display="inline",
                size="large",
                url=f"data:image/png;base64,{wind_data['image_base64']}"
            )
        )    
    # await cl.Message(
    #             content=output,
    #             elements=elements,
    #             author="Sailors Parrot"
    #         ).send()
    
    # what is return here goes to the model that used the tool
    # response_for_llm = transform(response)
    return {
        "grib_file": wind_data['grib_file'],
        "data_points": wind_data['data_points'],
        "elements": elements,
    }

@tool
async def process_documents() -> List[Document]:
    """Retrieve relevant sailing and boating documents from the knowledge base.
    Use this tool when the user is asking about sailing techniques, boat maintenance, or general sailing knowledge."""
    session = cl.user_session.get("session")
    result = await documents_tool(session.model)

    return result

def create_tool_calling_agent():
    # Create prompt for the agent
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful sailing and boating assistant. You can:
1. Get current wind/weather data for specific locations
2. Answer questions about sailing techniques, boat maintenance, and general sailing knowledge

Choose the most appropriate tool based on the question's intent and content."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Create LLM
    # todo might not need to pass a temperature here
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    # Create the agent with tools
    tools = [process_wind_data, process_documents]
    llm_with_tools = llm.bind_tools(tools)
    # agent = create_tool_calling_agent(llm, tools, prompt)
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return llm_with_tools


async def process_query(state):
    """Process the query using the agent."""
    try:
        llm_with_tools = cl.user_session.get("llm_with_tools")

        messages = [
            HumanMessage(state.question),
        ]

        result = await llm_with_tools.ainvoke(messages)
        
        if 'content' in result:
             await cl.Message(
                content=result['content'],
                author="Sailors Parrot"
            ).send()
        else:     
            # Execute each tool call
            # todo handle no tool_calls
            for tool_call in result.tool_calls:
                selected_tool = {
                    "process_wind_data": process_wind_data, 
                    "process_documents": process_documents}[tool_call["name"].lower()]
                
                tool_result = await selected_tool.ainvoke(tool_call["args"])

            elements = tool_result.get('elements', [])
            content = tool_result.get('content', '')

            if elements or content:
                await cl.Message(
                    elements=elements,
                    content=content,
                    author="Sailors Parrot"
                ).send()
            
    except Exception as e:
        logger.error(f"Error in agent execution: {e}")
        return {"next": "reject"}
        


@cl.on_chat_start
async def on_chat_start():
    try:
        # Initialize session ID when chat starts
        cl.user_session.set("session_id", str(uuid.uuid4()))
        
        # Initialize the session manager with the State class
        session = SessionManager(State)
        cl.user_session.set("session", session)
        
        llm_with_tools = create_tool_calling_agent()

        cl.user_session.set("llm_with_tools", llm_with_tools)
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
            
        # Create state for this retrieval process
        # TODO I think this should go into the state as the query
        # and why 'model' and not 'state'? Oh.. it is the model.  We call it state everywhere else.
        session.model.question = message.content
        
        # Create a message object for streaming that we'll use internally
        # temp_msg = cl.Message(content="", author="Sailors Parrot")
        # await temp_msg.send()  # Send the empty message so we can stream to it
        message = cl.Message(content="", author="Sailors Parrot")
        session.model.current_message = message
        
        result = await process_query(session.model)
        
        # Batch updates
        with session.batch_update() as state:
            # TODO need to totally redo chat history
            # state.chat_history = result["chat_history"]
            # state.chat_history.append({"role": "human", "content": message.content})
            # state.chat_history.append({"role": "assistant", "content": result["answer"]})
            state.current_message = None
        
        # Add sources if needed
        # settings = cl.user_session.get("settings", UserSettings())
        # if settings.show_sources and result.get("current_context"):
        #     temp_msg.elements = format_docs_as_sources(result["current_context"])
        #     await temp_msg.update()

        # TODO: we are returning the message here but it was already sent
        # in process_query.  Process query should be returning what is needed
        # to build the message
        return message
        
    except Exception as e:
        logger.error("Error in message processing", exc_info=True)
        await handle_error(e, "processing")

@cl.on_chat_end
async def end():
    try:
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

