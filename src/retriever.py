from langchain_core.documents import Document
from typing_extensions import Literal, get_args
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Define forum topics as a type
ForumTopic = Literal[
    "Anchoring & Mooring",
    "Auxiliary Equipment & Dinghy",
    "Boat Ownership & Making a Living",
    "Construction, Maintenance & Refit",
    "Cooking and Provisioning: Food & Drink",
    "Deck hardware: Rigging, Sails & Hoisting",
    "Electrical: Batteries, Generators & Solar",
    "Engines and Propulsion Systems",
    "Families, Kids and Pets Afloat",
    "Fishing, Recreation & Fun",
    "General Sailing Forum",
    "Health, Safety & Related Gear",
    "Lithium Power Systems",
    "Liveaboard's Forum",
    "Marine Electronics",
    "Monohull Sailboats",
    "Multihull Sailboats",
    "Navigation",
    "OpenCPN",
    "Plumbing Systems and Fixtures",
    "Powered Boats",
    "Product or Service Reviews & Evaluations",
    "Propellers & Drive Systems",
    "Rules of the Road, Regulations & Red Tape",
    "Seamanship & Boat Handling",
    "Training, Licensing & Certification"
]

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
    Ye be a wise old sea captain with decades of sailing experience. 
    Your task is to help landlubbers and new sailors understand the ways of the sea using the following context.
    
    Previous conversation:
    {chat_history}
    
    Here's the relevant information from the ship's logs:
    {context}
    
    Remember:
    1. Speak like a seasoned captain - use nautical terms but explain them
    2. Be patient and detailed - these are new sailors asking
    3. If the users question doesn't make sense, ask them to clarify it.  Do not just make up an answer.
    4. If ye find multiple solutions, list them all
    5. If the context isn't sufficient, admit it
    6. Format your response using these markdown guidelines:
      - Use #### for main headings (smaller and cleaner)
      - Use ##### for subsections
      - Use - for bullet points
      - Use *italics* for nautical terms
      - Use **bold** for important points
      - Use > for tips and explanations
      - Keep paragraphs short and well-spaced
"""

class Search(BaseModel):
    """Search query with metadata filters."""
    query: str = Field(description="Search query to run")
    topics: ForumTopic = Field(description="Topic to search within")

    @classmethod
    def default(cls) -> "Search":
        return cls(
            query="",
            topics="General Sailing Forum"
        )

class State(BaseModel):
    question: str
    query: Search = Field(default_factory=Search.default)
    current_context: List[Document] = Field(default_factory=list)  # Documents for current query
    running_context: List[Document] = Field(default_factory=list)  # Accumulated context
    answer: str = Field(default="")
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

@dataclass
class RetrieverConfig:
    """Configuration for the Retriever"""
    metadata_search_k: int = 15    # Number of docs to retrieve with metadata filter
    semantic_search_k: int = 15    # Number of docs to retrieve with semantic search
    doc_window_size: int = 30      # Maximum number of docs to keep in context window
    chat_window_size: int = 10     # Number of message pairs to keep in chat history
    query_template: str = ""
    system_template: str = ""
    chat: dict = Field(default_factory=dict)

# Create a single instance of the config with formatted templates
CONFIG = RetrieverConfig(
    query_template=BASE_QUERY_TEMPLATE % {
        "topics": "\n".join(f"- {topic}" for topic in FORUM_TOPICS_LIST)
    },
    system_template=BASE_SYSTEM_TEMPLATE
)

class RetrievalError(Exception):
    """Custom error for retrieval failures"""
    pass

class Retriever:
    """Handles document retrieval and response generation."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseChatModel
    ):
        self.vector_store = vector_store
        self.llm = llm
        
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

    async def analyze_query(self, state: State) -> Dict[str, Search]:
        """Generate optimized search query with metadata filters."""
        try:
            logger.info("Analyzing query")
            structured_llm = self.llm.with_structured_output(Search)
            query = await structured_llm.ainvoke(
                self.query_prompt.format(question=state.question)
            )
            logger.info(f"Generated search query: {query}")
            return {"query": query}
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}", exc_info=True)
            return {
                "query": {
                    "query": state.question,
                    "topics": "General Sailing Forum"
                }
            }

    async def retrieve(self, state: State) -> Dict[str, List[Document]]:
        """Retrieve relevant documents using both metadata filtering and semantic search."""
        try:
            logger.info(f"Retrieving documents for query: {state.query}")
            
            # Get documents using metadata filter
            metadata_docs = await self.vector_store.asimilarity_search(
                state.query.query,
                k=CONFIG.metadata_search_k,
                filter={"topics": state.query.topics}
            )
            logger.info(f"Retrieved {len(metadata_docs)} metadata filtered docs")
            
            # Get documents using pure semantic search
            semantic_docs = await self.vector_store.asimilarity_search(
                state.query.query,
                k=CONFIG.semantic_search_k
            )
            logger.info(f"Retrieved {len(semantic_docs)} semantic search docs")
            
            # Combine results, removing duplicates while preserving order
            seen_ids = set()
            combined_docs = []
            
            # Add metadata filtered docs first
            for doc in metadata_docs:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    combined_docs.append(doc)
            
            # Add unique semantic docs
            for doc in semantic_docs:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    combined_docs.append(doc)
            
            logger.info(f"Combined into {len(combined_docs)} unique documents")
            
            # Update state with retrieved documents
            # state.current_context = combined_docs
            logger.info(f"Updated state.current_context with {len(combined_docs)} docs")
            
            return {
                "current_context": combined_docs
                }
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}")

    async def generate(self, state: State) -> Dict[str, str]:
        """Generate response using both current and running context."""
        try:
            logger.info("Generating response from context")
            logger.info(f"Initial state - current_context: {len(state.current_context)} docs, running_context: {len(state.running_context)} docs")
            
            logger.info(f"State current_context length: {len(state.current_context)}")
            logger.info(f"State running_context length: {len(state.running_context)}")
            
            # Remove docs from running context that are in current context
            current_ids = {doc.id for doc in state.current_context}
            filtered_running_docs = [
                doc for doc in state.running_context 
                if doc.id not in current_ids
            ]
            logger.info(f"Removed {len(state.running_context) - len(filtered_running_docs)} duplicate docs from running context")
            logger.info(f"Filtered running docs: {len(filtered_running_docs)}")
            
            # Add current docs at front, running docs at back
            combined_docs = state.current_context + filtered_running_docs
            logger.info(f"Combined docs before trimming: {len(combined_docs)}")
            
            # Trim to window size
            combined_docs = combined_docs[:CONFIG.doc_window_size]
            logger.info(f"Document window size after trimming: {len(combined_docs)} docs")
            
            # Update running context
            # state.running_context = combined_docs
            # logger.info(f"Updated running context size: {len(state.running_context)}")
            
            # Trim chat history if needed
            # Todo: this code should be in the app.py file
            chat_history = state.chat_history
            if len(chat_history) > CONFIG.chat_window_size * 2:  # Keep pairs of messages
                chat_history = chat_history[-CONFIG.chat_window_size * 2:]
            
            # Generate response using combined context
            docs_content = "\n\n".join(doc.page_content for doc in combined_docs)
            
            # Format messages using prompt template
            messages = await self.prompt.ainvoke({
                "question": state.question,
                "context": docs_content,
                "chat_history": state.chat_history
            })
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            logger.info("Response generated successfully")
            
            return {
                "answer": response.content,
                "running_context": combined_docs,
                "chat_history": chat_history
                }
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {"answer": "Apologies, but I encountered an error while trying to answer your question."} 