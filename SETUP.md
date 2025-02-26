# Platform-Adaptive Setup Guide

This document explains how the application has been configured to run efficiently on both Apple Silicon (M1/M2) machines and GPU-enabled systems like RunPod.

## Architecture Overview

The application has been designed to automatically detect the platform it's running on and use the appropriate implementation:

1. **Apple Silicon / CPU-only mode**: Uses a custom implementation with memory-mapped tensors and CPU-only processing
2. **GPU mode**: Uses the standard implementation with CUDA acceleration

## Key Components

### 1. IndexProvider

The `IndexProvider` class is responsible for:
- Detecting the platform type
- Loading the appropriate implementation of `RAGMultiModalModel`
- Managing the shared index instance
- Providing a clean interface for the rest of the application

```python
# IndexProvider automatically selects the appropriate implementation
from visual_index.index_provider import IndexProvider

# Get the index - platform-specific details are handled internally
index = IndexProvider.get_index("visual_books")
```

### 2. Custom Implementation for Apple Silicon

For Apple Silicon, we use a custom implementation that:
- Forces CPU usage to avoid CUDA initialization errors
- Uses memory-mapped tensors for efficient memory usage
- Applies monkey patches to prevent CUDA-related errors

```python
# This is handled automatically by the IndexProvider
# You don't need to import these directly
from custom_modules.byaldi import RAGMultiModalModel  # For Apple Silicon
```

### 3. Standard Implementation for GPU Systems

For systems with NVIDIA GPUs, we use the standard Byaldi implementation:
```python
# This is handled automatically by the IndexProvider
# You don't need to import these directly
from byaldi.RAGModel import RAGMultiModalModel  # For GPU systems
```

### 4. Docker Configurations

Two Docker configurations are provided:

1. **Standard Docker (CPU mode)**: `docker-compose.yml`
   - Uses the CPU-optimized implementation
   - Suitable for Apple Silicon and other CPU-only environments

2. **GPU Docker**: `docker-compose.gpu.yml`
   - Uses NVIDIA CUDA for acceleration
   - Suitable for RunPod and other GPU-enabled environments

## Running the Application

### Checking Your Environment

Run the environment checker script to determine the best configuration for your system:

```bash
python check_environment.py
```

This will analyze your system and recommend the appropriate Docker configuration.

### On Apple Silicon (or CPU-only systems)

```bash
docker-compose up -d
```

### On GPU Systems (RunPod or other NVIDIA GPU environments)

```bash
docker-compose -f docker-compose.gpu.yml up -d
```

## Memory Considerations

- **Apple Silicon**: The application uses memory-mapped tensors to efficiently manage memory usage
- **GPU Systems**: The application leverages GPU memory for faster processing

## Troubleshooting

### Apple Silicon Issues

If you encounter memory issues on Apple Silicon:
1. Increase the Docker memory limit in `docker-compose.yml`
2. Consider using a smaller model by changing `VISUAL_MODEL_NAME` in the environment variables

### GPU Issues

If you encounter GPU issues:
1. Ensure NVIDIA drivers are properly installed
2. Check that nvidia-docker is configured correctly
3. Verify that CUDA is available to PyTorch 