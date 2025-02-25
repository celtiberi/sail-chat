"""
Centralized configuration for the application.
This module contains all configuration classes and settings used throughout the application.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from typing_extensions import Literal, get_args
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
if not load_dotenv():
    print("Warning: No .env file found")

# Get workspace root
WORKSPACE_ROOT = Path(__file__).parent.parent.absolute()

# ======================
# LLM Configuration
# ======================

@dataclass
class LLMConfig:
    """Language Model Configuration"""
    model_name: str = "gemini-2.0-flash"  # Model to use
    temperature: float = 0.7  # Higher = more creative, Lower = more focused
    top_p: float = 0.8  # Nucleus sampling parameter
    max_tokens: Optional[int] = None  # Max output length (None = model default)

# ======================
# Chat Configuration
# ======================

@dataclass
class ChatConfig:
    """Chat History and Context Configuration"""
    max_history_items: int = 20  # Maximum number of messages to keep
    context_window: int = 5  # Number of recent messages for context

# ======================
# Retriever Configuration
# ======================

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

@dataclass
class RetrieverConfig:
    """Configuration for the Retriever"""
    metadata_search_k: int = 15    # Number of docs to retrieve with metadata filter
    semantic_search_k: int = 15    # Number of docs to retrieve with semantic search
    visual_doc_search_k: int = 5   # Number of visual docs to retrieve
    doc_window_size: int = 30      # Maximum number of docs to keep in context window
    chat_window_size: int = 10     # Number of message pairs to keep in chat history
    query_template: str = ""
    system_template: str = ""
    chat: dict = Field(default_factory=dict)
    forum_collection: str = "forum_content"
    num_forum_results: int = 10
    
    # ChromaDB configuration
    local_db_path: str = "./chroma_db"
    
    def get_db_path(self) -> str:
        """Get absolute path to ChromaDB directory"""
        return os.path.abspath(self.local_db_path)

# ======================
# Visual Search Configuration
# ======================

@dataclass
class VisualSearchConfig:
    """Configuration for visual search functionality"""
    DPI: int = 300
    CHUNK_SIZE: int = 10
    MODEL_NAME: str = "vidore/colqwen2-v1.0"
    DEVICE: str = "cpu"
    IMAGE_FORMAT: str = "png"
    INDEX_ROOT: Path = WORKSPACE_ROOT / '.byaldi'
    FILTER_MULTIPLIER: int = 3  # Get 3x results when filtering to ensure enough after filtering

# ======================
# User Settings
# ======================

@dataclass
class UserSettings:
    """User-configurable settings"""
    show_sources: bool = False
    show_stats: bool = True
    debug_mode: bool = False

# ======================
# Main Application Configuration
# ======================

@dataclass
class AppConfig:
    """Main Application Configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    
    def validate(self):
        """Validate configuration values."""
        if self.chat.max_history_items < 1:
            raise ValueError("max_history_items must be positive")
        if not self.llm.model_name:
            raise ValueError("model_name must be specified")
        if self.llm.temperature < 0 or self.llm.temperature > 1:
            raise ValueError("temperature must be between 0 and 1")

# ======================
# Create Configuration Instances
# ======================

# Import forum topics for retriever config
try:
    from models import ForumTopic
    FORUM_TOPICS_LIST = list(get_args(ForumTopic))
except ImportError:
    FORUM_TOPICS_LIST = ["General Sailing Forum"]

# Create retriever config with formatted templates
RETRIEVER_CONFIG = RetrieverConfig(
    query_template=BASE_QUERY_TEMPLATE % {
        "topics": "\n".join(f"- {topic}" for topic in FORUM_TOPICS_LIST)
    },
    system_template=BASE_SYSTEM_TEMPLATE
)

# Create main app config
APP_CONFIG = AppConfig()

# Create visual search config
VISUAL_CONFIG = VisualSearchConfig()

# ======================
# Environment Variable Validation
# ======================

# Validate required environment variables
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set. "
        "Please set it in your .env file"
    )

# Override config values from environment variables
if visual_model := os.getenv("VISUAL_MODEL_NAME"):
    VISUAL_CONFIG.MODEL_NAME = visual_model

if visual_device := os.getenv("VISUAL_DEVICE"):
    VISUAL_CONFIG.DEVICE = visual_device

if forum_collection := os.getenv("FORUM_COLLECTION"):
    RETRIEVER_CONFIG.forum_collection = forum_collection 