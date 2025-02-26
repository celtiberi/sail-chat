from pydantic import BaseModel, Field
from typing_extensions import Literal, get_args
from langchain_core.documents import Document
from typing import List, Dict, Optional, Any

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
    question: str | None = None
    query: Search | None = Field(default_factory=Search.default)
    is_sailing_related: bool | None = None
    answer: str | None = None
    current_context: List[Document] | None = Field(default_factory=list)  # Documents for current query
    running_context: List[Document] | None = Field(default_factory=list)  # Accumulated context
    chat_history: List[Dict] | None = Field(default_factory=list)
    context: List[Document] | None = Field(default_factory=list)
    visual_context: List[Document] = Field(default_factory=list)  # Visual search results
    visual_files: List[str] = Field(default_factory=list)  # Paths to visual results
    current_step: Any = None  # Chainlit step for streaming responses
    current_message: Any = None  # Chainlit message for streaming responses
    steps: Dict[str, Any] = Field(default_factory=dict)  # Dictionary of steps for each node
    updated_steps: Dict[str, bool] = Field(default_factory=dict)  # Track which steps have been updated