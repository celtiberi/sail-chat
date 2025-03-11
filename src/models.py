from enum import Enum
from pydantic import BaseModel, Field
from typing_extensions import Literal, get_args
from langchain_core.documents import Document
from typing import List, Dict, Optional, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from src.config import CONFIG

# Configure logging
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

# User settings model
class UserSettings(BaseModel):
    show_sources: bool = True
    forum_topic: str = ""
    
# Message model for structured chat history
class Message(BaseModel):
    role: str
    content: str

class Search(BaseModel):
    """Search query with metadata filters."""
    query: str = Field(description="Search query to run")
    topics: str = Field(description="Topic to search within", default="")

    @classmethod
    def default(cls) -> "Search":
        return cls(
            query="",
            topics=""
        )
    
class State(BaseModel):
    question: str | None = None
    query: Search | None = Field(default_factory=Search.default)
    is_sailing_related: bool | None = None
    answer: str | None = None
    forum_docs: List[Document] = Field(default_factory=list)  # Forum search results
    visual_docs: List[Document] = Field(default_factory=list)  # Visual search results
    chat_history: List[Dict] | None = Field(default_factory=list)
    context: List[Document] | None = Field(default_factory=list)
    current_step: Any = None  # Chainlit step for streaming responses
    current_message: Any = None  # Chainlit message for streaming responses
    steps: Dict[str, Any] = Field(default_factory=dict)  # Dictionary of steps for each node
    updated_steps: Dict[str, bool] = Field(default_factory=dict)  # Track which steps have been updated

class SessionManager:
    """
    Manages session-specific resources and state.
    """
    
    def __init__(self):
        """
        Initialize the session manager.
        """
        # No need to initialize ChromaDB here as it's now handled by the service
        pass
    
    def get_forum_store(self):
        """
        Get the forum content store.
        
        Note: This method is kept for backward compatibility,
        but now returns None as the forum store is handled by the service.
        """
        return None