# Sailors Parrot: Multimodal Sailing Assistant

A sophisticated AI-powered chatbot designed to help new sailors understand the ways of the sea. This assistant combines the wisdom of an experienced sea captain with modern retrieval-augmented generation (RAG) technology and multimodal capabilities to provide comprehensive, accurate information about sailing and boating.

## Features

- **Multimodal Interaction**: Process both text queries and images to provide comprehensive sailing advice
- **Retrieval-Augmented Generation**: Pulls information from sailing forums and reference materials
- **Visual Search**: Analyze sailing-related images to provide context-specific information
- **Nautical Persona**: Responds with the voice and expertise of a seasoned sea captain
- **Markdown Formatting**: Delivers well-structured, easy-to-read responses
- **Conversation Memory**: Maintains context throughout the conversation
- **Forum Topic Classification**: Automatically categorizes queries by relevant sailing topics
- **Platform Adaptability**: Optimized for both Apple Silicon and GPU environments

## Architecture

The system is built on a modern RAG architecture with these key components:

- **Retriever**: Core component that handles document retrieval and response generation
- **Visual Index**: Processes and indexes images for visual search capabilities
- **LangChain Integration**: Leverages LangChain for prompt templates and LLM interactions
- **Multimodal LLM**: Uses advanced language models capable of processing both text and images
- **Platform Abstraction**: Automatically adapts to different hardware environments

For more details on the architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

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

# Install custom Byaldi module
pip install -e custom_modules/byaldi

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

### Starting the Application

```
chainlit run src/app.py
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
│       ├── search.py         # Visual search implementation
│       └── index_provider.py # Platform-adaptive index management
├── custom_modules/           # Custom extensions
│   └── byaldi/               # Custom multimodal RAG implementation
├── .env                      # Environment variables
├── Dockerfile                # CPU-optimized Docker configuration
├── Dockerfile.gpu            # GPU-optimized Docker configuration
├── docker-compose.yml        # Docker Compose for CPU environments
├── docker-compose.gpu.yml    # Docker Compose for GPU environments
├── ARCHITECTURE.md           # Detailed architecture documentation
├── CHANGES.md                # Summary of platform adaptability changes
├── GPU_SETUP.md              # GPU-specific setup instructions
└── README.md                 # This file
```

### Adding New Features

To extend the assistant's capabilities:

1. Add new forum topics in `models.py` (ForumTopic)
2. Enhance the retriever with additional data sources
3. Improve the visual search capabilities
4. Customize the system prompt for different use cases

## Docker Deployment

This application can be run in a Docker container for easier deployment and consistency across environments. We provide two different Docker configurations:

1. **CPU Mode** (for Apple Silicon and other CPU-only environments)
2. **GPU Mode** (for environments with NVIDIA GPUs)

### Prerequisites

- Docker and Docker Compose installed on your system
- Google API key for the language model
- For GPU mode: NVIDIA GPU with CUDA support and NVIDIA Container Toolkit installed

### Running on Apple Silicon (or CPU-only systems)

```bash
# Build and start the container in CPU mode
docker-compose up -d
```

This will:
- Build the Docker image optimized for CPU usage
- Mount the data directories as volumes
- Start the application on port 8000

### Running on GPU Systems

```bash
# Build and start the container in GPU mode
docker-compose -f docker-compose.gpu.yml up -d
```

This will:
- Build the Docker image optimized for GPU usage
- Configure the container to use NVIDIA GPUs
- Mount the data directories as volumes
- Start the application on port 8000

For detailed GPU setup instructions, see [GPU_SETUP.md](GPU_SETUP.md).

### Accessing the Application

Access the application at http://localhost:8000

### Environment Variables

You can set environment variables in the following ways:
- In a `.env` file in the project root (for docker-compose)
- Directly in the `docker-compose.yml` file
- By passing them to the `docker-compose up` command:

```bash
GOOGLE_API_KEY=your_key_here CHAINLIT_AUTH_SECRET=your_secret docker-compose up -d
```

### Data Persistence

The Docker setup uses volume mounts to persist data:
- `./data:/app/data` - Application data
- `./.byaldi:/app/.byaldi` - Byaldi index
- `./chroma_db:/app/chroma_db` - ChromaDB data

This ensures that your data remains intact even if the container is removed.

## Platform-Specific Optimizations

### Apple Silicon / CPU-only Environments

The application includes specific optimizations for Apple Silicon:
- Uses memory-mapped tensors for efficient memory usage
- Forces CPU usage to avoid CUDA initialization errors
- Applies custom patches to optimize for Apple Silicon

### GPU Environments

For systems with NVIDIA GPUs, the application:
- Uses CUDA acceleration for faster processing
- Configures memory allocation settings for optimal performance
- Leverages GPU memory for model inference

## Checking Your Environment

Run the environment checker script to determine the best configuration for your system:

```bash
python check_environment.py
```

This will analyze your system and recommend the appropriate Docker configuration.

## License

[Specify your license here]

## Contributing

[Guidelines for contributing to the project] 