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
            # Group documents by their main result (using doc_id)
            page_groups = {}
            
            # First, identify main pages and create groups
            for doc in state.visual_docs:
                if not doc.metadata.get('is_adjacent', False):
                    doc_id = doc.metadata.get('doc_id')
                    if doc_id:
                        # Create a new group with the main page
                        page_groups[doc_id] = {
                            'main': doc,
                            'adjacent': []
                        }
            
            # Then, add adjacent pages to their respective groups
            for doc in state.visual_docs:
                if doc.metadata.get('is_adjacent', False):
                    # Find the main page this is adjacent to
                    main_doc_id = doc.metadata.get('doc_id')
                    if main_doc_id and main_doc_id in page_groups:
                        page_groups[main_doc_id]['adjacent'].append(doc)
            
            # Create a structured description of the visual content
            visual_descriptions = []
            
            for group_id, group in page_groups.items():
                main_doc = group['main']
                adjacent_docs = group['adjacent']
                
                # Get book and page info
                book_title = main_doc.metadata.get('pdf_title', 'Unknown document')
                main_page = main_doc.metadata.get('page_num', 'unknown')
                
                # Start with the main page description
                group_desc = f"Document: {book_title}\n"
                group_desc += f"Main result - Page {main_page}: {main_doc.page_content}\n"
                
                # Add adjacent pages in order
                if adjacent_docs:
                    # Sort adjacent pages by page number
                    adjacent_docs.sort(key=lambda doc: int(doc.metadata.get('page_num', 0)))
                    
                    # List the page sequence
                    page_numbers = [main_page] + [doc.metadata.get('page_num', 'unknown') for doc in adjacent_docs]
                    
                    # Convert to integers and sort
                    sorted_page_numbers = [int(p) for p in page_numbers]
                    sorted_page_numbers.sort()
                    
                    group_desc += f"This result includes a sequence of pages: {', '.join(str(p) for p in sorted_page_numbers)}\n"
                    
                    # Add each adjacent page description
                    for adj_doc in adjacent_docs:
                        relation = adj_doc.metadata.get('relation', 'adjacent')
                        page = adj_doc.metadata.get('page_num', 'unknown')
                        group_desc += f"{relation.capitalize()} page {page}: {adj_doc.page_content}\n"
                
                visual_descriptions.append(group_desc)
            
            # Join all group descriptions
            all_descriptions = "\n\n".join(visual_descriptions)
            visual_content.append({"type": "text", "text": all_descriptions})
            
            # Add each image as a separate content item in the proper format
            for doc in state.visual_docs:
                if img_base64 := doc.metadata.get('image_base64'):
                    # Add page info to help identify the image
                    page_info = f"Page {doc.metadata.get('page', 'unknown')}"
                    if doc.metadata.get('is_adjacent', False):
                        page_info += f" ({doc.metadata.get('relation', 'adjacent')})"
                    
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

    async def _load_image_from_path(self, path_str: str, max_size: int = 512) -> str:
        """
        Load, resize, and encode an image from a path.
        
        Args:
            path_str: Path to the image file
            max_size: Maximum width or height of the image in pixels (default: 512)
            
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
                    
                    if width > max_size or height > max_size:
                        # Calculate the scaling factor
                        scale_factor = max_size / max(width, height)
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
    
    async def _get_adjacent_pages(self, path_str: str, metadata: Dict) -> List[Tuple[str, Dict]]:
        """
        Get paths to adjacent pages based on the current page.
        
        Args:
            path_str: Path to the current page
            metadata: Metadata for the current page
            
        Returns:
            List of tuples containing path and metadata for adjacent pages
        """
        adjacent_pages = []
        
        try:
            # Extract page number from the filename
            # Format: '/Users/patrickcremin/repo/chat/data/pdfs/0/images/page_0001.png'
            filename = os.path.basename(path_str)
            
            # Extract the page number using regex
            page_match = re.search(r'page_(\d+)\.png', filename)
            if page_match:
                current_page = int(page_match.group(1))
                logger.info(f"Extracted page number {current_page} from filename {filename}")
                
                # Get the directory path
                directory = os.path.dirname(path_str)
                
                # Create paths for previous and next pages
                for offset in [-1, 1]:  # -1 for previous page, 1 for next page
                    adjacent_page = current_page + offset
                    if adjacent_page > 0:  # Ensure page number is positive
                        # Format the page number with leading zeros to match the pattern
                        page_str = f"{adjacent_page:04d}"  # 4-digit with leading zeros
                        adj_filename = f"page_{page_str}.png"
                        adj_path = os.path.join(directory, adj_filename)
                        
                        # Check if the adjacent page file exists
                        if os.path.exists(adj_path):
                            # Create metadata for adjacent page
                            adj_metadata = metadata.copy()
                            adj_metadata['page_num'] = adjacent_page
                            adj_metadata['is_adjacent'] = True
                            adj_metadata['relation'] = 'previous' if offset == -1 else 'next'
                            
                            logger.info(f"Found {adj_metadata['relation']} page: {adj_path}")
                            adjacent_pages.append((adj_path, adj_metadata))
                        else:
                            logger.info(f"Adjacent page file does not exist: {adj_path}")
            else:
                logger.warning(f"Could not extract page number from filename: {filename}")
        except Exception as e:
            logger.error(f"Error getting adjacent pages for {path_str}: {e}")
        
        return adjacent_pages

    async def retrieve(self, state: State, max_image_size: int = 800) -> Dict[str, List[Document]]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            state: The current state containing the query.
            include_adjacent_pages: Whether to include adjacent pages in visual results (default: True)
            max_image_size: Maximum width or height of the images in pixels (default: 800)
            
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
                visual_docs = await self._retrieve_visual_docs(
                    state, 
                    include_adjacent_pages=False,
                    max_image_size=max_image_size
                )
            
                # Update step with the result
                retrieve_step.output = f"Retrieved {len(forum_docs) + len(visual_docs)} documents"
                await retrieve_step.update()
            
            # await retrieve_step.remove()
            # logger.info(f"Retrieve step removed")

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
    
    async def _retrieve_visual_docs(self, state: State, include_adjacent_pages: bool = True, max_image_size: int = 800) -> List[Document]:
        """
        Retrieve visual documents based on the query.
        
        Args:
            state: The current state containing the query.
            include_adjacent_pages: Whether to include adjacent pages in the results (default: True)
            max_image_size: Maximum width or height of the images in pixels (default: 800)
            
        Returns:
            A list of visual documents.
        """
        visual_docs = []
        
        # Search for visual documents
        try:
            logger.info(f"Searching for visual documents with query: {state.query}")
            
            # Use the service client to search for visual documents
            results, paths = await self.corpus_client.visual_search(
                query=state.query.query,
                k=CONFIG.corpus_visual_search_k,
            )
            # Update paths to include full data directory path
            paths = [str(CONFIG.data_dir / 'pdfs' / path) for path in paths]

            if results:
                logger.info(f"Found {len(results)} visual documents")
                
                # Process each visual search result
                for i, (result, path_str) in enumerate(zip(results, paths)):
                    
                    # Extract metadata from the result
                    metadata = result.metadata or {}
                    
                    # Create base metadata for the main page
                    main_metadata = {
                        "type": "visual",
                        "doc_id": result.doc_id,
                        "score": result.score,
                        "is_adjacent": False,  # This is the main page, not an adjacent one
                    }
                    
                    # Add any additional metadata from the result to the main page
                    if isinstance(metadata, dict):
                        main_metadata.update(metadata)
                    
                    # Get all pages (main and adjacent)
                    all_pages = [(path_str, main_metadata)]
                    
                    # Get adjacent pages and add them to the list if requested
                    if include_adjacent_pages:
                        adjacent_pages = await self._get_adjacent_pages(path_str, main_metadata)
                        all_pages.extend(adjacent_pages)
                    
                    # Create document for each page
                    for page_path, page_metadata in all_pages:
                        # Load the image with the specified max size
                        img_base64 = await self._load_image_from_path(page_path, max_size=max_image_size)
                        
                        # Only proceed if we successfully loaded the image
                        if img_base64:
                            # Create a description based on whether this is a main or adjacent page
                            if page_metadata.get('is_adjacent', False):
                                relation = page_metadata.get('relation', 'adjacent')
                                page_num = page_metadata.get('page_num', 'unknown')
                                
                                description = (
                                    f"{relation.capitalize()} page (page {page_num}) to the main result.\n"
                                )
                                
                                # Add book info if available
                                if 'pdf_title' in page_metadata:
                                    description += f"Book: {page_metadata.get('pdf_title', 'Unknown')}\n"
                            else:
                                # Create a rich description for the main image
                                description = f"Visual search result with confidence score: {page_metadata.get('score', 0.0):.2f}\n"
                                
                                # Add book info if available
                                if 'pdf_title' in page_metadata:
                                    description += f"Book: {page_metadata.get('pdf_title', 'Unknown')}\n"                                   
                                
                                # Add page info if available
                                if 'page_num' in page_metadata:
                                    description += f"Page: {page_metadata.get('page_num', 'Unknown')}\n"
                            
                            # Add the source path to metadata
                            page_metadata['source'] = path_str
                            
                            # Add the base64 image to metadata
                            page_metadata['image_base64'] = img_base64
                            
                            # Create the document
                            doc = Document(
                                page_content=description,
                                metadata=page_metadata
                            )
                            
                            # Append to visual_docs
                            visual_docs.append(doc)
                            
                            # Log the addition
                            if page_metadata.get('is_adjacent', False):
                                logger.info(f"Added {page_metadata.get('relation', 'adjacent')} page to visual_docs")
                            else:
                                logger.info(f"Added main visual result to visual_docs")
                        else:
                            logger.warning(f"Could not load image for path: {page_path}")
                
            else:
                logger.info("No visual documents found")
        except Exception as e:
            logger.error(f"Error searching for visual documents: {e}")
            
        
        return visual_docs

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