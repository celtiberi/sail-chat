import time
from langchain_core.documents import Document
from typing_extensions import Literal, get_args
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompt_values import PromptValue
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
import logging
from dataclasses import dataclass
from models import ForumTopic, State, Search
from services.corpus_service import CorpusClient
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
from openai import AsyncOpenAI
import re
import uuid

# Import centralized configuration
from src.config import RETRIEVER_CONFIG as CONFIG, WORKSPACE_ROOT, APP_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# Get list of topics from the Literal type
FORUM_TOPICS_LIST = list(get_args(ForumTopic))


class RetrievalError(Exception):
    """Custom error for retrieval failures"""
    pass

class Retriever:
    """Handles document retrieval and response generation."""
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the retriever and create the LLM.
        
        Args:
            embedding_model: Name of the embedding model (only used for logging)
            llm: Optional language model to use. If not provided, one will be created.
        """
        # Create or use the provided LLM
        logger.info(f"Creating LLM: {APP_CONFIG.llm.model_name} with max_tokens: {APP_CONFIG.llm.max_tokens}")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=APP_CONFIG.llm.temperature,
            # max_output_tokens=APP_CONFIG.llm.max_tokens,
            # top_p=APP_CONFIG.llm.top_p,
            streaming=True,
        )

        logger.info(f"Creating LLM: {APP_CONFIG.llm.model_name} with max_tokens: {APP_CONFIG.llm.max_tokens}")
        self.deepseek_client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        # Log the actual configuration used
        logger.info(f"LLM Configuration: model={APP_CONFIG.llm.model_name}, "
                    f"temperature={APP_CONFIG.llm.temperature}, "
                    f"max_output_tokens={APP_CONFIG.llm.max_tokens}, "
                    f"top_p={APP_CONFIG.llm.top_p}")
        
        # Initialize the service client
        services_url = os.getenv("SERVICES_URL", "http://localhost:8081")
        self.corpus_client = CorpusClient(services_url)
        logger.info(f"Initialized CorpusClient with URL: {services_url}")
        
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
            
            structured_llm = self.llm.with_structured_output(Search)

            with cl.Step(name="Analyze Query", type="tool") as analyze_step:
                query = await structured_llm.ainvoke(
                    self.query_prompt.format(question=state.question)
                )
            
                # Update step with the result
                analyze_step.output = f"Generated search query: {query.query}"
                await analyze_step.update()
                

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

    async def _prepare_forum_content(self, state: State) -> str:
        """
        Prepare forum documents by filtering, combining with running context, and formatting.
        
        Args:
            state: The current state object
            
        Returns:
            Formatted document content as a string
        """
        # Generate text content from documents
        docs_content = "\n\n".join(doc.page_content for doc in state.forum_docs)
        
        return docs_content
    
    async def _prepare_visual_message_items(self, state: State) -> List[Dict]:
        """
        Prepare visual content items for the user message.
        
        Args:
            state: The current state object
            
        Returns:
            List of visual content items to add to the user message
        """
        visual_content = []
        
        # Add visual content if available
        if state.visual_docs:
            # Create a structured description of the visual content
            visual_descriptions = []
            
            for doc in state.visual_docs:
                # Get book and page info
                book_title = doc.metadata.get('pdf_title', 'Unknown document')
                page_num = doc.metadata.get('page_num', 'unknown')
                
                # Create the description
                desc = f"Document: {book_title}\n"
                desc += f"Page {page_num}: {doc.page_content}\n"
                visual_descriptions.append(desc)
            
            # Join all descriptions
            all_descriptions = "\n\n".join(visual_descriptions)
            visual_content.append({"type": "text", "text": all_descriptions})
            
            # Add each image
            for doc in state.visual_docs:
                if img_base64 := doc.metadata.get('image_base64'):
                    # Add page info to help identify the image
                    page_info = f"Page {doc.metadata.get('page_num', 'unknown')}"
                    
                    # Format as data URL with proper MIME type
                    visual_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        },
                        "metadata": {
                            "page_info": page_info
                        }
                    })
        
        return visual_content

    async def _load_image_from_path(self, path_str: str) -> str:
        """
        Load, resize, and encode an image from a path.
        
        Args:
            path_str: Path to the image file
            
        Returns:
            Base64-encoded image string or None if loading failed
        """
        img_base64 = None
        
        try:
            logger.info(f"Attempting to load image from: {path_str}")
            
            # Check if the file exists
            if not os.path.exists(path_str):
                logger.warning(f"File does not exist: {path_str}")
                return None
                
            # Get original file size
            original_file_size = os.path.getsize(path_str)
            original_file_size_mb = original_file_size / (1024 * 1024)
            logger.info(f"Original file size: {original_file_size_mb:.2f} MB ({original_file_size} bytes)")
            
            # Load and encode the image if the file exists
            if os.path.exists(path_str):
                with Image.open(path_str) as img:
                    # Resize the image while maintaining aspect ratio
                    width, height = img.size
                    original_dimensions = f"{width}x{height}"
                    
                    if width > CONFIG.max_image_size or height > CONFIG.max_image_size:
                        # Calculate the scaling factor
                        scale_factor = CONFIG.max_image_size / max(width, height)
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        
                        # Resize the image
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        logger.info(f"Resized image from {original_dimensions} to {new_width}x{new_height}")
                    else:
                        logger.info(f"Image dimensions {original_dimensions} are within limits, no resizing needed")
                    
                    # Convert image to bytes with optimization for PNG
                    img_byte_arr = BytesIO()
                    img.save(img_byte_arr, format='PNG', optimize=True, compress_level=6)
                    img_byte_arr_value = img_byte_arr.getvalue()
                    
                    # Calculate compressed size
                    compressed_size = len(img_byte_arr_value)
                    compressed_size_mb = compressed_size / (1024 * 1024)
                    
                    # Calculate reduction percentage
                    reduction_percentage = ((original_file_size - compressed_size) / original_file_size) * 100
                    
                    logger.info(f"Compressed file size: {compressed_size_mb:.2f} MB ({compressed_size} bytes)")
                    logger.info(f"Size reduction: {reduction_percentage:.1f}% ({(original_file_size - compressed_size) / (1024 * 1024):.2f} MB saved)")
                    
                    # Convert to base64
                    img_base64 = base64.b64encode(img_byte_arr_value).decode('utf-8')
                    base64_size_mb = len(img_base64) / (1024 * 1024)
                    logger.info(f"Base64 encoded size: {base64_size_mb:.2f} MB")
                    logger.info(f"Successfully loaded and encoded image: {path_str}")
            else:
                logger.warning(f"Skipping image encoding as file does not exist: {path_str}")
        except Exception as e:
            logger.error(f"Error loading image {path_str}: {e}")
        
        return img_base64
    
    async def _retrieve_visual_docs(self, state: State) -> List[Document]:
        """
        Retrieve visual documents based on the query.
        
        Args:
            state: The current state containing the query.
            
        Returns:
            A list of visual documents.
        """
        visual_docs = []
        
        try:
            logger.info(f"Searching for visual documents with query: {state.query}")
            
            # Use the service client to search for visual documents
            results, paths = await self.corpus_client.visual_search(
                query=state.query.query,
                k=CONFIG.corpus_visual_search_k,
            )
            # Update paths to include full data directory path
            paths = [str(CONFIG.data_dir / 'pdfs' / path) for path in paths]

            if not results:
                logger.info("No visual documents found")
                return visual_docs

            logger.info(f"Found {len(results)} visual documents")
            
            # Prepare all tasks for parallel processing
            processing_tasks = []
            
            for i, (result, path_str) in enumerate(zip(results, paths)):
                # Extract metadata from the result
                metadata = result.metadata or {}
                
                # Create base metadata for the main page
                main_metadata = {
                    "type": "visual",
                    "doc_id": result.doc_id,
                    "score": result.score,
                }
                
                # Add any additional metadata from the result
                if isinstance(metadata, dict):
                    main_metadata.update(metadata)
                
                # Create task for processing the page
                task = self._process_single_page(path_str, main_metadata)
                processing_tasks.append(task)
            
            # Process all pages in parallel
            processed_docs = await asyncio.gather(*processing_tasks)
            
            # Filter out None results and add valid documents to visual_docs
            visual_docs = [doc for doc in processed_docs if doc is not None]
            
            logger.info(f"Successfully processed {len(visual_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error searching for visual documents: {e}")
        
        return visual_docs

    async def _process_single_page(self, page_path: str, page_metadata: Dict) -> Optional[Document]:
        """
        Process a single page, including loading and preparing the document.
        
        Args:
            page_path: Path to the image file
            page_metadata: Metadata for the page
            
        Returns:
            Document object if successful, None if failed
        """
        try:
            # Load the image with the specified max size
            img_base64 = await self._load_image_from_path(page_path)
            
            if not img_base64:
                logger.warning(f"Could not load image for path: {page_path}")
                return None
            
            # Create a rich description for the image
            description = f"Visual search result with confidence score: {page_metadata.get('score', 0.0):.2f}\n"
            
            # Add book info if available
            if 'pdf_title' in page_metadata:
                description += f"Book: {page_metadata.get('pdf_title', 'Unknown')}\n"
            
            # Add page info if available
            if 'page_num' in page_metadata:
                description += f"Page: {page_metadata.get('page_num', 'Unknown')}\n"
            
            # Add the source path and base64 image to metadata
            page_metadata['source'] = page_path
            page_metadata['image_base64'] = img_base64
            
            # Create and return the document
            doc = Document(
                page_content=description,
                metadata=page_metadata
            )
            
            logger.info(f"Added visual result for page {page_metadata.get('page_num', 'unknown')}")
            return doc
            
        except Exception as e:
            logger.error(f"Error processing page {page_path}: {e}")
            return None

    async def _prepare_gemini_messages(self, 
                                      state: State, 
                                      formatted_chat_history:  Optional[str] = '',
                                      docs_content: Optional[str] = '', 
                                      visual_content: Optional[List[Dict]] = None, 
                                      custom_prompt: Optional[str] = None):
        """
        Prepare messages for the Gemini model.
        
        Args:
            state: The current state containing the question
            docs_content: Text content from retrieved documents
            visual_content: List of visual content items (images, etc.)
            formatted_chat_history: Formatted chat history string
            custom_prompt: Optional custom system prompt (if None, uses CONFIG.system_template)
            
        Returns:
            Formatted messages ready to be sent to the Gemini model
        """
        user_message_content = [
            {"type": "text", "text": state.question}
        ]
        
        # Add visual content if available
        if visual_content:
            user_message_content.extend(visual_content)
        
        # Use custom prompt if provided, otherwise use the default
        system_prompt = custom_prompt if custom_prompt else CONFIG.system_template
        
        # Format the system prompt with chat history and context
        formatted_system_prompt = system_prompt.format(
            chat_history=formatted_chat_history,
            context=docs_content
        )
        
        # Create a modified prompt template that can handle multimodal content
        multimodal_prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_system_prompt),
            ("user", user_message_content)
        ])
        
        # Format the prompt to get the messages
        messages = await multimodal_prompt.ainvoke({})
        
        return messages

    async def generate(self, state: State) -> Dict[str, str]:
        """Generate response using both current and running context."""
        try:
            logger.info("Generating response from context")            
            
            # Trim chat history if needed
            chat_history = state.chat_history if state.chat_history else []
            if len(chat_history) > CONFIG.chat_window_size * 2:
                chat_history = chat_history[-CONFIG.chat_window_size * 2:]
            
            # Format chat history for the template
            formatted_chat_history = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in chat_history
            ])

            # Prepare content
            docs_content = await self._prepare_forum_content(state)                        
            visual_content = await self._prepare_visual_message_items(state)

            if APP_CONFIG.llm.model_name == "gemini-2.0-flash":
                # Initialize an empty response
                full_response = ""
                
                # Stream the response to the current message                
                async for chunk in self.google_flash_stream(
                    state=state, 
                    docs_content=docs_content, 
                    visual_content=visual_content, 
                    formatted_chat_history=formatted_chat_history):
                    # Stream to the message for visibility during generation
                    await state.current_message.stream_token(chunk)
                    # Collect the full response
                    full_response += chunk
                
                response_content = full_response
            
            elif APP_CONFIG.llm.model_name == "deepseek":
                # Initialize an empty response
                full_response = ""
                source_index = 1
                logger.info("Generating flash context")
                with cl.Step(name="Generate Context", type="tool") as retrieve_step:
                    # Prepare all context generation tasks
                    task_mapping = {}  # Map tasks to their types
                    tasks = []
                    # Add docs context task if we have docs_content
                    if docs_content:
                        docs_task = self.google_flash_complete(
                            state=state, 
                            docs_content=docs_content, 
                            visual_content=None, 
                            formatted_chat_history="",#formatted_chat_history,
                            custom_prompt=CONFIG.context_template
                        )
                        tasks.append(docs_task)
                    
                    # Add tasks for each visual item
                    for visual_item in visual_content:
                        image_task = self.google_flash_complete(
                            state=state,              
                            formatted_chat_history="",
                            docs_content="",
                            visual_content=[visual_item],
                            custom_prompt=CONFIG.context_template
                        )
                        tasks.append(image_task)
                    
                    # Process results as they complete
                    response_content = ""
                    current_index = 1
                    
                    # Process results as they arrive
                    for task in asyncio.as_completed(tasks):
                        result = await task
                        
                        with cl.Step(name=f"Source {current_index}", type="tool") as source_step:
                            source_step.output = result
                            await source_step.update()
                            response_content += f"\n\nSource {current_index}:\n{result}"
                        current_index += 1

                # Final update to the main retrieve step
                retrieve_step.output = response_content
                await retrieve_step.update()

                logger.info("Context generation completed")

                # Stream the response to the current message
                async for chunk in self.deepseek_stream(
                    state=state, 
                    docs_content=docs_content, 
                    visual_content=visual_content, 
                    formatted_chat_history=formatted_chat_history,
                    flash_response=response_content):
                    # Stream to the message for visibility during generation
                    await state.current_message.stream_token(chunk)
                    # Collect the full response
                    full_response += chunk
                
                # Send the message after streaming is complete
                await state.current_message.send()
                
                response_content = full_response

            return {
                "answer": response_content,
                "chat_history": chat_history
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"answer": "Apologies, but I encountered an error while trying to answer your question."} 

    async def google_flash_stream(self, 
                                  state: State, 
                                  docs_content: str, 
                                  visual_content: List[Dict], 
                                  formatted_chat_history: str,                                   
                                  custom_prompt: Optional[str] = None):
        """
        Generate a response using Google's Gemini model with streaming.
        
        Args:
            state: The current state containing the question
            docs_content: Text content from retrieved documents
            visual_content: List of visual content items (images, etc.)
            formatted_chat_history: Formatted chat history string
            custom_prompt: Optional custom system prompt (if None, uses CONFIG.system_template)
            
        Returns:
            An async generator yielding content chunks
        """
        # Get formatted messages
        messages = await self._prepare_gemini_messages(
            state, 
            docs_content, 
            visual_content, 
            formatted_chat_history,
            custom_prompt
        )
        
        # Get the stream
        response_stream = self.llm.astream(messages)
    
        # Stream the response and yield each chunk
        async for chunk in response_stream:
            if chunk.content:
                # Yield the chunk for the caller to handle
                yield chunk.content
    
        logger.info("Response streaming completed")
    
    async def google_flash_complete(self, 
                                   state: State, 
                                   docs_content: str, 
                                   formatted_chat_history: str, 
                                   visual_content: Optional[List[Dict]] = None, 
                                   custom_prompt: Optional[str] = None):
        """
        Generate a complete response using Google's Gemini model without streaming.
        
        Args:
            state: The current state containing the question
            docs_content: Text content from retrieved documents
            visual_content: List of visual content items (images, etc.)
            formatted_chat_history: Formatted chat history string
            custom_prompt: Optional custom system prompt (if None, uses CONFIG.system_template)
            
        Returns:
            The complete response as a string
        """
        # Get formatted messages
        messages = await self._prepare_gemini_messages(
            state, 
            formatted_chat_history,
            docs_content, 
            visual_content, 
            custom_prompt
        )
        
        response = await self.llm.ainvoke(messages)
        full_response = response.content
        
        logger.info("Response generation completed")
        
        # Return the full response
        return full_response

    async def deepseek_stream(self, 
                              state: State, 
                              docs_content: str, 
                              visual_content: List[Dict], 
                              formatted_chat_history: str,
                              flash_response: str):
        """
        Generate a response using DeepSeek's model with reasoning capabilities.
        
        Args:
            state: The current state containing the question
            docs_content: Text content from retrieved documents
            visual_content: List of visual content items (images, etc.)
            formatted_chat_history: Formatted chat history string
            
        Returns:
            An async generator yielding content chunks
        """
        # Call deepseek with a prompt that it is generating a response for AI consumption
        start = time.time()

        # Prepare system message with context
        system_prompt = CONFIG.system_template
        system_content = system_prompt.format(
            chat_history=formatted_chat_history,
            context=flash_response
        )

        # user_message_content = [
        #     {"type": "text", "text": state.question}
        # ]
        # todo need max_tokens 8k, defaults to 4k
        stream = await self.deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": state.question}
            ],
            stream=True
        )

        # Flag to track if we've exited the thinking step
        thinking_completed = False
        
        # Streaming the thinking
        async with cl.Step(name="Thinking") as thinking_step:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                reasoning_content = getattr(delta, "reasoning_content", None)
                if reasoning_content is not None and not thinking_completed:
                    await thinking_step.stream_token(reasoning_content)
                elif not thinking_completed:
                    # Exit the thinking step
                    thought_for = round(time.time() - start)
                    thinking_step.name = f"Thought for {thought_for}s"
                    await thinking_step.update()
                    thinking_completed = True
                    break
        
        # Streaming the final answer
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                # Yield the content chunk for the caller to handle
                yield delta.content

    async def check_service_health(self) -> bool:
        """
        Check if the service is healthy.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            is_healthy = await self.corpus_client.health_check()
            if is_healthy:
                logger.info("Service is healthy")
            else:
                logger.warning("Service is not healthy")
            return is_healthy
        except Exception as e:
            logger.error(f"Error checking service health: {e}")
            return False

    async def retrieve(self, state: State) -> Dict[str, List[Document]]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            state: The current state containing the query.
            
        Returns:
            A dictionary with forum_docs and visual_docs.
        """
        if not state.query:
            raise ValueError("No query provided")
        
        try:
            with cl.Step(name="Load Data", type="tool") as retrieve_step:
                # Get forum documents
                forum_docs = await self._retrieve_forum_docs(state)
                
                # Get visual documents
                visual_docs = await self._retrieve_visual_docs(state)
            
                # Update step with the result
                retrieve_step.output = f"Retrieved {len(forum_docs) + len(visual_docs)} documents"
                await retrieve_step.update()

            return {
                "forum_docs": forum_docs,
                "visual_docs": visual_docs,
            }
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            retrieve_step.error = str(e)
            retrieve_step.status = "error"
            raise RetrievalError(f"Error retrieving documents: {e}")
    
    async def _retrieve_forum_docs(self, state: State) -> List[Document]:
        """
        Retrieve forum documents based on the query.
        
        Args:
            state: The current state containing the query.
            
        Returns:
            A list of forum documents.
        """
        forum_docs = []
        
        # Search for documents in forum content
        try:
            logger.info(f"Searching for forum documents with query: {state.query}")
            
            # Use the service client to search for documents
            results = await self.corpus_client.chroma_search(
                query=state.query.query,
                k=CONFIG.corpus_chroma_search_k,
                filter={"topics": state.query.topics},
            )
            
            if results:
                logger.info(f"Found {len(results)} forum documents")
                forum_docs = results
            else:
                logger.info("No forum documents found")
        except Exception as e:
            logger.error(f"Error searching for forum documents: {e}")
            # Continue with empty forum_docs
        
        return forum_docs