from langchain_core.documents import Document
from typing_extensions import Literal, get_args
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
import logging
from dataclasses import dataclass
from models import ForumTopic, State, Search
from services.client import ServiceClient
from PIL import Image
from io import BytesIO
import base64
import os
from langchain_chroma import Chroma
from pathlib import Path
import asyncio
import chainlit as cl
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

# Import centralized configuration
from src.config import RETRIEVER_CONFIG as CONFIG, WORKSPACE_ROOT, APP_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# Get list of topics from the Literal type
FORUM_TOPICS_LIST = list(get_args(ForumTopic))

# Define base templates
BASE_QUERY_TEMPLATE = """Given a question, create a search query that will help find relevant information to answer it.
    Focus on extracting key terms and concepts. Also determine which forum topic would be most relevant.
    
    Available topics:
    %(topics)s
    
    Question: {question}
    
    Return a search query and the most relevant topic from the list above."""

BASE_SYSTEM_TEMPLATE = """
    You are a wise sea captain with decades of sailing and boating experience. 
    Your task is to help new sailors understand the ways of the sea using the following context.
    
    Previous conversation:
    {chat_history}
    
    Here's the relevant information from various sailing/boating sources:
    {context}
    
    Remember:
    1. Use nautical terms but explain them
    2. Be patient and detailed - these are new sailors asking
    3. If the users question doesn't make sense, ask them to clarify it.  Do not just make up an answer.
    4. If you find multiple solutions, list them all
    5. If the context isn't sufficient, answer as best as you can and admit it.  This is important. We do not want to give them bad advice for the dangerous seas of boating.
    6. Refrain from talking about users or the forums.  Just stick to the information.
    7. Do not use phrases like "The text mentions".  Talk as though all of the information is yours so that there is no confusion.
    9. Do not use phrases like "described in AN114" or "waypoint AN1174".  The user cannot see what you are referring to so you must describe it.
    10. Do not quote passages from the context.  Just use the information to answer the question.
    11. When asked for help with a sailing route, give as much details as possible.  What to be wary of, prevailing winds, weather conditions, best times of year for the trip, and anything else that is important.
    11. Format your response using these markdown guidelines:
      - Use #### for main headings (smaller and cleaner)
      - Use ##### for subsections
      - Use - for bullet points
      - Use *italics* for nautical terms
      - Use **bold** for important points
      - Use > for tips and explanations
      - Keep paragraphs short and well-spaced
"""

class RetrievalError(Exception):
    """Custom error for retrieval failures"""
    pass

class Retriever:
    """Handles document retrieval and response generation."""
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        llm: Optional[BaseChatModel] = None
    ):
        """
        Initialize the retriever and create the LLM.
        
        Args:
            embedding_model: Name of the embedding model (only used for logging)
            llm: Optional language model to use. If not provided, one will be created.
        """
        # Create or use the provided LLM
        if llm is None:
            logger.info(f"Creating LLM: {APP_CONFIG.llm.model_name} with max_tokens: {APP_CONFIG.llm.max_tokens}")
            self.llm = ChatGoogleGenerativeAI(
                model=APP_CONFIG.llm.model_name,
                temperature=APP_CONFIG.llm.temperature,
                # max_output_tokens=APP_CONFIG.llm.max_tokens,
                # top_p=APP_CONFIG.llm.top_p,
                streaming=True,
            )
            # Log the actual configuration used
            logger.info(f"LLM Configuration: model={APP_CONFIG.llm.model_name}, "
                       f"temperature={APP_CONFIG.llm.temperature}, "
                       f"max_output_tokens={APP_CONFIG.llm.max_tokens}, "
                       f"top_p={APP_CONFIG.llm.top_p}")
        else:
            self.llm = llm
            logger.info("Using provided LLM")
        
        # Initialize the service client
        services_url = os.getenv("SERVICES_URL", "http://localhost:8081")
        self.service_client = ServiceClient(services_url)
        logger.info(f"Initialized ServiceClient with URL: {services_url}")
        
        # Create query analysis prompt
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", CONFIG.query_template),
            ("human", "{question}")
        ])
        
        # Create prompt template with structured chat history
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CONFIG.system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

    def reject_query(self, state: State) -> Dict:
        """Handle non-sailing queries."""
        # The validate step should already be created and stored in the state
        # We just need to update it with the rejection message
        if hasattr(state, 'steps') and 'reject' in state.steps:
            step = state.steps['reject']
            step.output = "Query rejected as it's not sailing-related"
            
            # Update the step
            asyncio.create_task(step.update())
            
            # Remove the step after a short delay
            async def remove_step_after_delay():
                await asyncio.sleep(1)  # Short delay to let users see the rejection
                await step.remove()
                logger.info("Reject step removed")
            
            asyncio.create_task(remove_step_after_delay())
            
            # Mark the step as updated
            if hasattr(state, 'updated_steps'):
                state.updated_steps['reject'] = True
        
        response = "I can only answer questions about boating and maritime topics. Please rephrase your question to focus on boating-related matters."
        return {"answer": response}

    async def analyze_query(self, state: State) -> Dict[str, Search]:
        """Generate optimized search query with metadata filters."""
        try:
            logger.info("Analyzing query")
            
            # Create and send the analyze step without ephemeral parameter
            analyze_step = cl.Step(name="Analyze Query", type="tool", show_input=True)
            analyze_step.input = state.question
            await analyze_step.send()
            
            # Store the step in the state
            state.steps["analyze"] = analyze_step
            
            # Update the analyze step
            analyze_step.output = "Analyzing query to generate optimized search terms..."
            await analyze_step.update()
            
            structured_llm = self.llm.with_structured_output(Search)
            query = await structured_llm.ainvoke(
                self.query_prompt.format(question=state.question)
            )
            
            # Update step with the result
            analyze_step.output = f"Generated search query: {query.query} in topic: {query.topics}"
            await analyze_step.update()
                
            # Mark the step as updated
            if hasattr(state, 'updated_steps'):
                state.updated_steps['analyze'] = True
            
            await analyze_step.remove()
            logger.info(f"Analyze step removed")

            logger.info(f"Generated search query: {query}")
            return {
                "query": query
                }
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}", exc_info=True)
            return {
                "query": {
                    "topics": ""
                }
            }

   

    async def retrieve(self, state: State) -> Dict[str, List[Document]]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            state: The current state object
            
        Returns:
            Dictionary with retrieved documents
        """
        try:
            # Get the query from the state
            query = state.query.query if state.query else state.question
            if not query:
                raise ValueError("No query provided")
            
            # Create and send the retrieve step without ephemeral parameter
            retrieve_step = cl.Step(name="Retrieve Documents", type="tool", show_input=True)
            retrieve_step.input = f"Query: {state.query.query}, Topic: {state.query.topics}"
            await retrieve_step.send()
            
            # Store the step in the state
            state.steps["retrieve"] = retrieve_step
            
            # Update the retrieve step
            retrieve_step.output = f"Retrieving documents for: {state.query.query}"
            await retrieve_step.update()

            # Get documents from forum content
            forum_docs = []
            try:
                # Search the forum content using the service
                logger.info(f"Searching ChromaDB with query: '{query}', filter: {filter}")
                forum_docs = await self.service_client.chroma_search(
                    query=query,
                    k=CONFIG.num_forum_results,
                    filter={"topics": state.query.topics}
                )
                
                logger.info(f"Retrieved {len(forum_docs)} documents from forum content")

            except Exception as e:
                logger.error(f"Error retrieving forum documents: {e}")
            
            # Get documents from visual search
            visual_docs = []
            visual_files = []
            try:
                # Use the service to search
                results, paths = await self.service_client.visual_search(
                    query=query,
                    k=5
                )
                
                # Convert paths to strings and store them
                string_paths = [str(path) for path in paths]
                
                # Process each visual search result
                for i, (result, path_str) in enumerate(zip(results, string_paths)):
                    # Initialize variables
                    full_path = ""
                    img_base64 = None
                    
                    try:
                        # Construct the full path by prepending CONFIG.data_dir / 'pdfs'
                        full_path = str(CONFIG.data_dir / 'pdfs' / path_str)
                        logger.info(f"Attempting to load image from: {full_path}")
                        
                        # Check if the file exists
                        if not os.path.exists(full_path):
                            logger.warning(f"File does not exist: {full_path}")
                            # Try alternative paths
                            alt_paths = [
                                str(WORKSPACE_ROOT / 'data' / 'pdfs' / path_str),
                                str(Path(path_str)),  # Try the path as-is
                                str(Path('data') / 'pdfs' / path_str)
                            ]
                            
                            # Try each alternative path
                            for alt_path in alt_paths:
                                logger.info(f"Trying alternative path: {alt_path}")
                                if os.path.exists(alt_path):
                                    full_path = alt_path
                                    logger.info(f"Using alternative path: {full_path}")
                                    break
                            else:
                                logger.warning(f"Could not find the image file in any of the tried paths")
                        
                        # Load and encode the image if the file exists
                        if os.path.exists(full_path):
                            with Image.open(full_path) as img:
                                # Convert image to bytes
                                img_byte_arr = BytesIO()
                                img.save(img_byte_arr, format='PNG')
                                img_byte_arr = img_byte_arr.getvalue()
                                # Convert to base64
                                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                                logger.info(f"Successfully loaded and encoded image: {full_path}")
                        else:
                            logger.warning(f"Skipping image encoding as file does not exist: {full_path}")
                    except Exception as e:
                        logger.error(f"Error loading image {full_path}: {e}")
                        img_base64 = None
                    
                    # Extract metadata from the result
                    metadata = result.metadata if hasattr(result, 'metadata') else {}
                    
                    # Create a rich description of the image
                    description = (
                        f"Visual result from {path_str} with confidence score {result.score:.2f}.\n"
                    )
                    
                    # Add additional metadata to the description if available
                    if isinstance(metadata, dict):
                        description += (
                            f"This image shows {metadata.get('description', 'a sailing-related scene')}.\n"
                        )
                    
                    # Create the document metadata
                    doc_metadata = {
                        "source": full_path,  # Use the full path in the metadata
                        "score": result.score,
                        "type": "visual",
                        "doc_id": result.doc_id
                    }
                    
                    # Add the base64 image to metadata if available
                    if img_base64:
                        doc_metadata['image_base64'] = img_base64
                    
                    # Add any additional metadata from the result
                    if isinstance(metadata, dict):
                        doc_metadata.update(metadata)
                    
                    # Create the document
                    doc = Document(
                        page_content=description,
                        metadata=doc_metadata
                    )
                    
                    visual_docs.append(doc)
                
                # Store the paths (use the full paths where possible)
                visual_files = []
                for path_str in string_paths:
                    full_path = str(CONFIG.data_dir / 'pdfs' / path_str)
                    if os.path.exists(full_path):
                        visual_files.append(full_path)
                    else:
                        # Try alternative paths
                        alt_path = str(WORKSPACE_ROOT / 'data' / 'pdfs' / path_str)
                        if os.path.exists(alt_path):
                            visual_files.append(alt_path)
                        else:
                            # Fall back to the original path
                            visual_files.append(str(path_str))
                
                logger.info(f"Retrieved {len(visual_docs)} documents from visual search")
            except Exception as e:
                logger.error(f"Error retrieving visual documents: {e}")
            
            # Combine all documents
            all_docs = forum_docs + visual_docs
            
            # Store the documents in the state
            state.current_context = all_docs
            state.visual_context = visual_docs
            state.visual_files = visual_files
            
            # Update step with the result
            retrieve_step.output = f"Retrieved {len(all_docs)} documents"
            await retrieve_step.update()
                
            # Mark the step as updated
            if hasattr(state, 'updated_steps'):
                state.updated_steps['retrieve'] = True
            
            await retrieve_step.remove()
            logger.info(f"Retrieve step removed")
            
            return {
                "current_context": all_docs,
                "visual_context": visual_docs,
                "visual_files": visual_files
            }
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise RetrievalError(f"Failed to retrieve documents: {e}")

    async def generate(self, state: State) -> Dict[str, str]:
        """
        Generate response using both current and running context.
        
        Args:
            state: The current state object
            
        Returns:
            Dictionary with the generated response and updated state
        """
        try:
            logger.info("Generating response from context")
            logger.info(f"Initial state - current_context: {len(state.current_context)} docs, running_context: {len(state.running_context if state.running_context else [])} docs")
            
             # Create and send the generate step without ephemeral parameter
            generate_step = cl.Step(name="Generate Response", type="run", show_input=False)
            await generate_step.send()
            
            # Store the step in the state
            state.steps["generate"] = generate_step
            
            # Update the generate step
            generate_step.output = "Generating response from retrieved documents..."
            await generate_step.update()
            
            # Remove docs from running context that are in current context
            current_ids = {getattr(doc, 'id', i) for i, doc in enumerate(state.current_context)}
            filtered_running_docs = []
            if state.running_context:
                filtered_running_docs = [
                    doc for i, doc in enumerate(state.running_context) 
                    if getattr(doc, 'id', i) not in current_ids
                ]
            
            # Add current docs at front, running docs at back
            combined_docs = state.current_context + filtered_running_docs
            
            # Trim to window size
            combined_docs = combined_docs[:CONFIG.doc_window_size]
            
            # Trim chat history if needed
            chat_history = state.chat_history if state.chat_history else []
            if len(chat_history) > CONFIG.chat_window_size * 2:
                chat_history = chat_history[-CONFIG.chat_window_size * 2:]
            
            # Format chat history for the template
            formatted_chat_history = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in chat_history
            ])
            
            # Generate text content from documents
            docs_content = "\n\n".join(doc.page_content for doc in combined_docs)
            
            # Create a list to hold message content (text + images)
            user_message_content = [
                # First add the text content
                {"type": "text", "text": state.question}
            ]
            
            # Add visual content if available
            if state.visual_context:
                # Add a text description of the visual content
                visual_descriptions = "\n\n".join(
                    f"Image {i+1} description: {doc.page_content}" 
                    for i, doc in enumerate(state.visual_context)
                )
                user_message_content.append({"type": "text", "text": visual_descriptions})
                
                # Add each image as a separate content item in the proper format
                for i, doc in enumerate(state.visual_context):
                    if img_base64 := doc.metadata.get('image_base64'):
                        # Format as data URL with proper MIME type
                        user_message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        })
            
            # Create a modified prompt template that can handle multimodal content
            multimodal_prompt = ChatPromptTemplate.from_messages([
                ("system", CONFIG.system_template.format(
                    chat_history=formatted_chat_history,
                    context=docs_content
                )),
                ("user", user_message_content)
            ])
            
            # Format the prompt to get the messages
            messages = await multimodal_prompt.ainvoke({})
            
            # Initialize an empty response
            full_response = ""
            
            # Get the stream
            stream = self.llm.astream(messages)
            
            # Flag to track if streaming has started
            streaming_started = False
            
            # Stream the response
            async for chunk in stream:
                if chunk.content:
                    # If this is the first chunk with content, update and remove the step
                    if not streaming_started:
                        streaming_started = True
                        logger.info("Streaming started, updating and removing generate step")
                        
                        # Final update to the step
                        generate_step.output = "Response generation started, streaming to message..."
                        await generate_step.update()
                        
                        # Remove the step now that streaming has started
                        await generate_step.remove()
                        logger.info("Generate step removed at start of streaming")
                    
                    # Stream to the message for visibility during generation                    
                    await state.current_message.stream_token(chunk.content)
                    
                    # Collect the full response
                    full_response += chunk.content
            
            logger.info("Response streaming completed")
            response_content = full_response
            
            logger.info("Response generated successfully")
            
            return {
                "answer": response_content,
                "running_context": combined_docs,
                "chat_history": chat_history,
                "visual_context": state.visual_context,
                "visual_files": state.visual_files
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"answer": "Apologies, but I encountered an error while trying to answer your question."} 

    
    async def check_service_health(self) -> bool:
        """
        Check if the service is healthy.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            is_healthy = await self.service_client.health_check()
            if is_healthy:
                logger.info("Service is healthy")
            else:
                logger.warning("Service is not healthy")
            return is_healthy
        except Exception as e:
            logger.error(f"Error checking service health: {e}")
            return False