from pydantic import BaseModel, Field
from typing_extensions import Literal, get_args
from langchain_core.documents import Document
from typing import List, Dict

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
    current_context: List[Document] | None = Field(default_factory=list)  # Documents for current query
    running_context: List[Document] | None = Field(default_factory=list)  # Accumulated context
    answer: str | None = Field(default="")
    chat_history: List[Dict[str, str]] | None = Field(default_factory=list)