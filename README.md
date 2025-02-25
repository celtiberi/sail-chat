# Sea Captain: Multimodal Sailing Assistant

A sophisticated AI-powered chatbot designed to help new sailors understand the ways of the sea. This assistant combines the wisdom of an experienced sea captain with modern retrieval-augmented generation (RAG) technology and multimodal capabilities to provide comprehensive, accurate information about sailing and boating.

## Features

- **Multimodal Interaction**: Process both text queries and images to provide comprehensive sailing advice
- **Retrieval-Augmented Generation**: Pulls information from sailing forums and reference materials
- **Visual Search**: Analyze sailing-related images to provide context-specific information
- **Nautical Persona**: Responds with the voice and expertise of a seasoned sea captain
- **Markdown Formatting**: Delivers well-structured, easy-to-read responses
- **Conversation Memory**: Maintains context throughout the conversation
- **Forum Topic Classification**: Automatically categorizes queries by relevant sailing topics

## Architecture

The system is built on a modern RAG architecture with these key components:

- **Retriever**: Core component that handles document retrieval and response generation
- **Visual Index**: Processes and indexes images for visual search capabilities
- **LangChain Integration**: Leverages LangChain for prompt templates and LLM interactions
- **Multimodal LLM**: Uses advanced language models capable of processing both text and images

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sea-captain.git
cd sea-captain

# Create and activate a conda environment
conda create -n sea-captain python=3.11
conda activate sea-captain

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

### Starting the Application

```bash
python src/app.py
```

### Example Queries

The assistant can handle a variety of sailing-related questions:

- "What's the best way to tie a bowline knot?"
- "How do I read a nautical chart?"
- "What should I do if I encounter rough weather while sailing?"
- [Upload an image of sailing equipment] "What is this and how do I use it?"
- "What safety equipment should I have on board for coastal cruising?"

## Configuration

The system is highly configurable through the `RetrieverConfig` class:

- Adjust retrieval parameters (number of documents, window sizes)
- Modify the system prompt and query templates
- Configure visual search parameters
- Set forum collection names and result limits

## Dependencies

- LangChain: For RAG implementation and LLM interaction
- ChromaDB: Vector database for semantic search
- PIL/Pillow: Image processing
- Byaldi: Custom multimodal RAG implementation
- Google Gemini or similar multimodal LLM

## Development

### Project Structure

```
.
├── src/
│   ├── app.py                # Main application entry point
│   ├── retriever.py          # Core retrieval and generation logic
│   ├── models.py             # Data models and state management
│   ├── session_manager.py    # User session handling
│   └── visual_index/         # Image processing and visual search
│       └── search.py         # Visual search implementation
├── custom_modules/           # Custom extensions
│   └── byaldi/               # Custom multimodal RAG implementation
├── .env                      # Environment variables
└── README.md                 # This file
```

### Adding New Features

To extend the assistant's capabilities:

1. Add new forum topics in `models.py` (ForumTopic)
2. Enhance the retriever with additional data sources
3. Improve the visual search capabilities
4. Customize the system prompt for different use cases

## License

[Specify your license here]

## Contributing

[Guidelines for contributing to the project] 