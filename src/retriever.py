from langchain_core.documents import Document
from typing_extensions import Literal, get_args
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
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
    
    Here's the relevant information from the ship's logs:
    {context}
    
    Remember:
    1. Speak like a seasoned captain - use nautical terms but explain them
    2. Be patient and detailed - these are new sailors asking
    3. If ye find multiple solutions, list them all
    4. If the context isn't sufficient, admit it
    5. Format your response using these markdown guidelines:
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
    context: List[Document] = Field(default_factory=list)
    answer: str = Field(default="")

@dataclass
class RetrieverConfig:
    """Configuration for the Retriever"""
    num_docs: int = 5  # Number of documents to retrieve
    max_tokens: int = 4096  # Max tokens for context window
    temperature: float = 0.7  # Temperature for response generation
    query_template: str = ""
    system_template: str = ""

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
        
        # Create QA prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CONFIG.system_template),
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
        """Retrieve relevant documents based on the question."""
        try:
            logger.info(f"Retrieving documents for query: {state.query}")
            retrieved_docs = await self.vector_store.asimilarity_search(
                state.query.query,
                k=CONFIG.num_docs
            )
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            return {"context": retrieved_docs}
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}")

    async def generate(self, state: State) -> Dict[str, str]:
        """Generate response using retrieved documents."""
        try:
            logger.info("Generating response from context")
            docs_content = "\n\n".join(doc.page_content for doc in state.context)
            
            # Format messages using prompt template
            messages = await self.prompt.ainvoke({
                "question": state.question,
                "context": docs_content
            })
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            logger.info("Response generated successfully")
            
            return {"answer": response.content}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {"answer": "Apologies, but I encountered an error while trying to answer your question."} 