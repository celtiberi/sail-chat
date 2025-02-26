# Changes for Platform Adaptability

This document summarizes the changes made to ensure the application runs efficiently on both Apple Silicon (M1/M2) and GPU-enabled environments.

## Recent Bug Fixes

### 1. Fixed Model Initialization Issue

Fixed an issue with the custom Byaldi module where:
- The `model_cls` parameter was being incorrectly passed to the parent class's `from_index` method
- Duplicate `device` parameters were being passed to the `CustomColPaliModel` constructor
- Import paths were updated to use absolute imports (`src.config` instead of `config`)

These changes ensure that the application can properly initialize the visual search index on Apple Silicon.

## Core Architecture Changes

### 1. IndexProvider Implementation

Created a new `IndexProvider` class (`src/visual_index/index_provider.py`) that:
- Abstracts platform-specific details from the rest of the application
- Detects the platform type automatically
- Loads the appropriate implementation of `RAGMultiModalModel`
- Manages a shared index instance with thread safety
- Provides a clean interface for accessing the index

### 2. Custom Byaldi Module for Apple Silicon

Modified `custom_modules/byaldi` to:
- Force CPU usage on Apple Silicon to avoid CUDA initialization errors
- Use memory-mapped tensors for efficient memory usage
- Apply monkey patches to prevent CUDA-related errors
- Provide a custom `RAGMultiModalModel` implementation optimized for Apple Silicon

### 3. Simplified Search Implementation

Updated `src/visual_index/search.py` to:
- Remove platform-specific logic and conditional imports
- Use the `IndexProvider` to abstract platform details
- Provide a clean interface for searching visual content

### 4. Configuration Updates

Modified `src/config.py` to:
- Simplify device selection logic
- Use "auto" as the default device setting
- Let the implementation decide the appropriate device based on the platform

## Docker Configuration

### 1. CPU Mode (Docker Compose)

Updated `docker-compose.yml` for:
- Apple Silicon and other CPU-only environments
- Memory-efficient operation with optimized settings

### 2. GPU Mode (Docker Compose GPU)

Created `docker-compose.gpu.yml` with:
- NVIDIA CUDA support
- GPU-specific environment variables
- Performance optimizations for GPU environments

### 3. Dockerfile Updates

Updated `Dockerfile.gpu` with:
- CUDA-specific environment variables
- Performance optimizations for GPU usage
- Proper NVIDIA runtime configuration

## User Tools and Documentation

### 1. Environment Checker

Created `check_environment.py` to:
- Analyze the user's system
- Detect platform type and available resources
- Recommend the appropriate Docker configuration
- Provide guidance on memory requirements

### 2. Container Management

Created `restart_containers.sh` to:
- Detect the system type
- Stop and remove existing containers
- Rebuild with the appropriate Docker configuration
- Provide a simple interface for managing containers

### 3. Documentation Updates

Updated documentation files:
- `README.md`: Added Docker deployment instructions for both environments
- `SETUP.md`: Created a platform-adaptive setup guide
- This `CHANGES.md` file to document all modifications

## Memory Optimization

### 1. Apple Silicon Optimizations

- Implemented memory-mapped tensor loading
- Forced CPU usage to avoid CUDA initialization errors
- Applied custom patches to optimize for Apple Silicon

### 2. GPU Optimizations

- Set appropriate CUDA environment variables
- Configured memory allocation settings
- Optimized for NVIDIA GPU performance

## Testing and Validation

The application has been tested on:
- Apple Silicon (M1/M2) machines
- NVIDIA GPU environments (including RunPod)
- Standard CPU-only environments

## Future Considerations

- Further memory optimizations for large models
- Additional platform-specific optimizations
- Performance monitoring and logging enhancements 