from langchain_core.documents import Document
from typing_extensions import Literal, get_args
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
import logging
from dataclasses import dataclass
from models import ForumTopic, State, Search
from visual_index.search import VisualSearch
import base64
from PIL import Image
from io import BytesIO
import os
from langchain_chroma import Chroma
from pathlib import Path
import asyncio
import chainlit as cl
from langchain_core.retrievers import BaseRetriever

# Import centralized configuration
from src.config import RETRIEVER_CONFIG as CONFIG, WORKSPACE_ROOT

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
        llm: BaseChatModel,
        embedding_model: str = "all-MiniLM-L6-v2",
        visual_search=None,  # Visual search instance
        forum_store=None     # ChromaDB instance
    ):
        """
        Initialize the retriever with LLM and optional shared resources.
        
        Args:
            llm: The language model to use for query analysis and response generation
            embedding_model: Name of the embedding model (only used if forum_store not provided)
            visual_search: Optional shared VisualSearch instance
            forum_store: Optional shared ChromaDB instance
        """
        self.llm = llm
        self.visual_search = visual_search
        
        # Use the provided forum_store or raise an error
        if forum_store:
            self.forum_store = forum_store
            logger.info("Using provided ChromaDB instance")
        else:
            logger.warning("No ChromaDB instance provided - retrieval functionality will be limited")
            self.forum_store = None
        
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
                    "topics": "General Sailing Forum"
                }
            }

    async def retrieve(self, state: State) -> Dict[str, List[Document]]:
        """Retrieve relevant documents from both forum and book collections."""
        try:
            logger.info("Retrieving query")
            logger.info(f"Retrieving documents for query: {state.query}")
            
            # Create and send the retrieve step without ephemeral parameter
            retrieve_step = cl.Step(name="Retrieve Documents", type="tool", show_input=True)
            retrieve_step.input = f"Query: {state.query.query}, Topic: {state.query.topics}"
            await retrieve_step.send()
            
            # Store the step in the state
            state.steps["retrieve"] = retrieve_step
            
            # Update the retrieve step
            retrieve_step.output = f"Retrieving documents for: {state.query.query}"
            await retrieve_step.update()
            
            # Get forum documents
            forum_docs = []
            if self.forum_store:
                forum_docs = await self.forum_store.asimilarity_search(
                    query=state.query.query,
                    k=CONFIG.num_forum_results,
                    filter={"topics": state.query.topics}
                )
                logger.info(f"Retrieved {len(forum_docs)} forum docs")
            else:
                logger.warning("Forum store not available - skipping forum retrieval")
            
            # Get visual search results if available
            visual_docs = []
            visual_files = []
            if self.visual_search:
                try:
                    results, raw_files = self.visual_search.search(
                        query=state.query.query,
                        k=CONFIG.visual_doc_search_k  # Use the configured value
                    )
                    # Rewrite paths to be relative to working directory
                    files = []
                    for file in raw_files:
                        # Split at data/pdfs and take the second part
                        relative_path = str(file).split("data/pdfs")[-1]
                        # Create new path from workspace root instead of current working directory
                        files.append(str(WORKSPACE_ROOT / "data/pdfs" / relative_path.lstrip("/")))
                    
                    # Convert results to Documents
                    for result, file in zip(results, files):
                        # Get additional metadata from Byaldi
                        metadata = self.visual_search.RAG.model.doc_id_to_metadata[result.doc_id][0]
                        
                        # Load and encode the image
                        try:
                            with Image.open(file) as img:
                                # Convert image to bytes
                                img_byte_arr = BytesIO()
                                img.save(img_byte_arr, format='PNG')
                                img_byte_arr = img_byte_arr.getvalue()
                                # Convert to base64
                                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                        except Exception as e:
                            logger.error(f"Error loading image {file}: {str(e)}")
                            img_base64 = None
                        
                        # Create a richer description of what was found
                        description = (
                            f"Visual result from {file} with confidence score {result.score:.2f}.\n"
                            f"This image shows {metadata.get('description', 'a sailing-related scene')}.\n"
                            f"Book: {metadata.get('book_title', 'Unknown')}\n"
                            f"Chapter: {metadata.get('chapter', 'N/A')}\n"
                            f"Page: {metadata.get('page', 'N/A')}"
                        )
                        
                        # Add base64 image to metadata if available
                        if img_base64:
                            metadata['image_base64'] = img_base64
                        
                        visual_docs.append(Document(
                            page_content=description,
                            metadata={
                                "source": str(file),
                                "score": result.score,
                                "type": "visual",
                                "doc_id": result.doc_id,
                                **metadata  # Include all additional metadata
                            }
                        ))
                    visual_files = files
                    logger.info(f"Retrieved {len(visual_docs)} visual results")
                except Exception as e:
                    logger.error(f"Error in visual search: {str(e)}", exc_info=True)
            else:
                logger.info("Visual search not available - skipping visual retrieval")
            

            
            # Combine results, removing duplicates while preserving order
            seen_ids = set()
            combined_docs = []
            
            # Add forum docs first
            for doc in forum_docs:
                doc_id = doc.id  # Will raise KeyError if id missing
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    combined_docs.append(doc)
            
            
            logger.info(f"Combined into {len(combined_docs)} unique documents")
            
            # Update step with the result
            retrieve_step.output = f"Retrieved {len(combined_docs)} documents"
            await retrieve_step.update()
                
            # Mark the step as updated
            if hasattr(state, 'updated_steps'):
                state.updated_steps['retrieve'] = True
            
            await retrieve_step.remove()
            logger.info(f"Retrieve step removed")

            return {
                "current_context": combined_docs,
                "visual_context": visual_docs,
                "visual_files": visual_files
            }
            
        except KeyError as e:
            logger.error("Document missing required id in metadata", exc_info=True)
            raise RetrievalError("Document integrity error: missing id in metadata")
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}")

    async def generate(self, state: State) -> Dict[str, str]:
        """Generate response using both current and running context."""
        try:
            logger.info("Generating response from context")
            logger.info(f"Initial state - current_context: {len(state.current_context)} docs, running_context: {len(state.running_context)} docs")
            
            # Create and send the generate step without ephemeral parameter
            generate_step = cl.Step(name="Generate Response", type="run", show_input=False)
            await generate_step.send()
            
            # Store the step in the state
            state.steps["generate"] = generate_step
            
            # Update the generate step
            generate_step.output = "Generating response from retrieved documents..."
            await generate_step.update()
            
            # Remove docs from running context that are in current context
            current_ids = {doc.id for doc in state.current_context}
            filtered_running_docs = [
                doc for doc in state.running_context 
                if doc.id not in current_ids
            ]
            
            # Add current docs at front, running docs at back
            combined_docs = state.current_context + filtered_running_docs
            
            # Trim to window size
            combined_docs = combined_docs[:CONFIG.doc_window_size]
            
            # Trim chat history if needed
            chat_history = state.chat_history
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
                        # Assuming JPEG format, adjust if needed
                        user_message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
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
            
            # Update generate step to show we're about to stream
            generate_step.output = "Preparing to stream response..."
            await generate_step.update()
                
            # Mark the step as updated
            if hasattr(state, 'updated_steps'):
                state.updated_steps['generate'] = True
            
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
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {"answer": "Apologies, but I encountered an error while trying to answer your question."} 