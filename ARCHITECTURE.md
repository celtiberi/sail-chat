# Platform-Adaptive Architecture

This document describes the architecture of the application, focusing on how it adapts to different platforms (Apple Silicon and GPU environments).

## Architecture Overview

The application has been designed with a platform-adaptive architecture that automatically detects the environment it's running in and uses the appropriate implementation:

```
┌───────────────────────────────────────────────────────────────┐
│                      Application Layer                         │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │    Retriever    │  │  Visual Search  │  │  Other Comps  │  │
│  └────────┬────────┘  └────────┬────────┘  └───────────────┘  │
└───────────┼─────────────────────┼─────────────────────────────┘
            │                     │
┌───────────▼─────────────────────▼─────────────────────────────┐
│                    Platform Abstraction                        │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │               IndexProvider                             │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │                       │
┌───────────────▼───────────┐ ┌─────────▼─────────────────┐
│  Apple Silicon / CPU Path │ │      GPU Path             │
│                           │ │                           │
│  ┌───────────────────┐    │ │  ┌───────────────────┐    │
│  │ Custom Byaldi     │    │ │  │ Standard Byaldi   │    │
│  │ Implementation    │    │ │  │ Implementation    │    │
│  └───────────────────┘    │ │  └───────────────────┘    │
└───────────────────────────┘ └───────────────────────────┘
```

## Key Components

### 1. IndexProvider

**Location**: `src/visual_index/index_provider.py`

**Purpose**: Abstracts platform-specific details and provides a unified interface for accessing the visual search index.

**Key Features**:
- Singleton pattern for managing a shared index instance
- Thread-safe access to the index
- Automatic platform detection
- Conditional loading of the appropriate implementation

**Usage Example**:
```python
from visual_index.index_provider import IndexProvider

# Get the index - platform-specific details are handled internally
index = IndexProvider.get_index("visual_books")
```

### 2. Visual Search

**Location**: `src/visual_index/search.py`

**Purpose**: Provides search functionality for visual content.

**Key Features**:
- Uses the IndexProvider to abstract platform details
- Implements search methods that work across platforms
- Handles filtering and result processing

**Usage Example**:
```python
from visual_index.search import VisualSearch

# Create a search instance
search = VisualSearch()

# Search for images similar to the query
results = search.search("path/to/image.jpg")
```

### 3. Custom Byaldi Implementation

**Location**: `custom_modules/byaldi/`

**Purpose**: Provides an optimized implementation for Apple Silicon and CPU-only environments.

**Key Files**:
- `__init__.py`: Entry point with platform detection and conditional logic
- `custom_colpali.py`: Custom implementation of the ColPaLI model with memory-mapped tensors

**Key Features**:
- Forces CPU usage to avoid CUDA initialization errors
- Uses memory-mapped tensors for efficient memory usage
- Applies monkey patches to prevent CUDA-related errors

### 4. Configuration

**Location**: `src/config.py`

**Purpose**: Centralizes configuration settings for the application.

**Key Features**:
- Uses "auto" as the default device setting
- Lets the implementation decide the appropriate device based on the platform
- Provides configuration classes for different components

## Docker Environment

The application can run in two Docker environments:

### 1. CPU Mode (`docker-compose.yml`)

**Purpose**: Optimized for Apple Silicon and other CPU-only environments.

**Key Features**:
- Uses the `continuumio/miniconda3:latest` base image
- Installs PyTorch CPU-only version
- Sets `CUDA_VISIBLE_DEVICES=-1` to force CPU usage
- Optimizes memory usage for CPU environments

**Configuration**:
```yaml
# Key environment variables in docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=-1
  - VISUAL_DEVICE=cpu
```

### 2. GPU Mode (`docker-compose.gpu.yml`)

**Purpose**: Optimized for NVIDIA GPU environments.

**Key Features**:
- Uses the `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` base image
- Installs PyTorch with CUDA support
- Configures NVIDIA runtime for Docker
- Sets GPU-specific environment variables

**Configuration**:
```yaml
# Key settings in docker-compose.gpu.yml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  - VISUAL_DEVICE=cuda
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Data Flow

1. **User Request**: The user submits a query through the application interface.

2. **Business Logic**: The application processes the query and determines if visual search is needed.

3. **Visual Search**: If visual search is required, the application calls the `VisualSearch` class.

4. **IndexProvider**: The `VisualSearch` class uses the `IndexProvider` to get the appropriate index.

5. **Platform-Specific Implementation**: The `IndexProvider` loads either:
   - The custom implementation for Apple Silicon / CPU-only environments
   - The standard implementation for GPU environments

6. **Result Processing**: The search results are processed and returned to the user.

## Memory Management

### Apple Silicon / CPU Path

- Uses memory-mapped tensors to efficiently manage memory
- Forces CPU usage to avoid CUDA initialization errors
- Optimizes for limited memory environments

### GPU Path

- Leverages GPU memory for faster processing
- Uses CUDA acceleration when available
- Optimizes for performance with appropriate memory settings

## Logging and Monitoring

The application includes comprehensive logging to track:
- Platform detection
- Implementation selection
- Device usage
- Memory usage
- Performance metrics

## Recent Bug Fixes

### Model Initialization Issue

Fixed an issue with the custom Byaldi module where:
- The `model_cls` parameter was being incorrectly passed to the parent class's `from_index` method
- Duplicate `device` parameters were being passed to the `CustomColPaliModel` constructor
- Import paths were updated to use absolute imports (`src.config` instead of `config`)

These changes ensure that the application can properly initialize the visual search index on Apple Silicon.

## Utility Scripts

### 1. Environment Checker (`check_environment.py`)

**Purpose**: Analyzes the system and recommends the appropriate Docker configuration.

**Key Features**:
- Detects platform type (Apple Silicon, Linux with GPU, etc.)
- Checks for NVIDIA GPU and CUDA availability
- Verifies Docker and NVIDIA Container Toolkit installation
- Provides recommendations based on the detected environment

### 2. Container Manager (`restart_containers.sh`)

**Purpose**: Simplifies container management with automatic platform detection.

**Key Features**:
- Automatically detects the system type
- Stops and removes existing containers
- Starts containers with the appropriate configuration
- Provides feedback and logs for troubleshooting

## Future Enhancements

Potential areas for future architectural improvements:
- Further abstraction of platform-specific components
- Dynamic model loading based on available resources
- Performance monitoring and auto-scaling
- Additional optimizations for specific hardware configurations
- Support for multi-GPU environments 