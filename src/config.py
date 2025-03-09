"""
Centralized configuration for the application.
This module contains all configuration classes and settings used throughout the application.
"""

import os
import platform
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

# Detect system type
is_apple_silicon = platform.system() == "Darwin" and platform.machine().startswith("arm")

# ======================
# LLM Configuration
# ======================

@dataclass
class LLMConfig:
    """Language Model Configuration"""
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gemini-2.0-flash"))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    top_p: float = field(default_factory=lambda: float(os.getenv("TOP_P", "0.8")))
    max_tokens: Optional[int] = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "0")) or None)  # Max output length (None = model default)

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
    Make sure your answer is as indepth as possible.  The user is asking for a lot of information.
    
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
    chat: dict = field(default_factory=dict)
    forum_collection: str = "forum_content"
    num_forum_results: int = 10
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))  # Path to data directory
    
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
    MODEL_NAME: str = "vidore/colqwen2-v1.0"
    # Device selection is handled by the platform-specific implementation
    # This is just a default that may be overridden
    DEVICE: str = "auto"  # Let the implementation decide based on platform
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

# Check if the data directory exists
data_dir = Path(RETRIEVER_CONFIG.data_dir)
if not data_dir.exists():
    raise ValueError(
        f"Data directory '{data_dir}' does not exist. "
        f"Please create the directory or set the DATA_DIR environment variable to a valid path."
    )

# Check if the pdfs subdirectory exists
pdfs_dir = data_dir / 'pdfs'
if not pdfs_dir.exists():
    raise ValueError(
        f"PDF directory '{pdfs_dir}' does not exist. "
        f"Please create the directory '{pdfs_dir}' to store PDF files."
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

class Config(BaseModel):
    """
    Application configuration with sensible defaults.
    Values can be overridden by environment variables.
    """
    
    # Base paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    
    # Database paths
    chroma_db_dir: Path = Field(default_factory=lambda: Path(os.getenv("CHROMA_DB_DIR", "chroma_db")))
    
    # Model configuration
    model_name: str = os.getenv("MODEL_NAME", "gemini-1.5-pro-latest")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "8000"))
    top_p: float = float(os.getenv("TOP_P", "0.95"))
    
    # Embedding model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Visual search configuration
    visual_search_url: str = os.getenv("VISUAL_SEARCH_URL", "http://localhost:8081")
    visual_index_dir: Path = Field(default_factory=lambda: Path(os.getenv("VISUAL_INDEX_DIR", ".byaldi")))
    
    # Server configuration
    port: int = int(os.getenv("CHAINLIT_PORT", "8080"))
    host: str = os.getenv("CHAINLIT_HOST", "0.0.0.0")
    
    # Prompt templates
    system_template: str = """
    You are a helpful nautical assistant specializing in sailing and boating knowledge.
    Use the provided context to answer the user's question. If you don't know the answer, say so - don't make up information.
    Keep your answers concise, accurate, and focused on the nautical domain.
    """
    
    query_template: str = """
    Analyze the following user question about sailing or boating:
    
    Question: {question}
    
    Determine the main topic, any specific terms to search for, and whether this is a question about:
    1. Sailing techniques
    2. Boat maintenance
    3. Navigation
    4. Safety
    5. Equipment
    6. Regulations
    7. Other (specify)
    
    Return your analysis in JSON format with the following fields:
    - main_topic: The primary topic of the question
    - search_terms: Key terms to search for (list)
    - category: One of the categories above
    - requires_visual: Whether visual information would help (boolean)
    """

# Create a global configuration instance
CONFIG = Config()

# Ensure directories exist
CONFIG.data_dir.mkdir(exist_ok=True, parents=True)
CONFIG.chroma_db_dir.mkdir(exist_ok=True, parents=True)
CONFIG.visual_index_dir.mkdir(exist_ok=True, parents=True) 