# GPU Environment Setup Guide

This guide provides instructions for setting up and running the application in a GPU-enabled environment.

## Prerequisites

- NVIDIA GPU with CUDA support
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

## Setup Instructions

1. **Verify NVIDIA drivers and CUDA installation**

   ```bash
   nvidia-smi
   ```

   This command should display information about your GPU and driver version.

2. **Verify NVIDIA Container Toolkit installation**

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```

   This command should display the same GPU information as the previous step, confirming that Docker can access your GPU.

3. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

4. **Configure environment variables**

   Create a `.env` file in the root directory with the following variables:

   ```
   GOOGLE_API_KEY=your_google_api_key
   CHAINLIT_AUTH_SECRET=your_secret_key_here
   ```

5. **Build and start the container**

   ```bash
   docker-compose -f docker-compose.gpu.yml up -d
   ```

   This will build the Docker image and start the container in detached mode.

6. **Check container logs**

   ```bash
   docker logs -f sailing-assistant-gpu
   ```

   You should see the application starting up and loading the visual search index.

7. **Access the application**

   Open your browser and navigate to:

   ```
   http://localhost:8000
   ```

## Troubleshooting

### GPU not detected

If the application is not using the GPU, check the following:

1. Verify that the NVIDIA Container Toolkit is properly installed:
   ```bash
   docker info | grep -i runtime
   ```
   You should see "nvidia" listed as one of the runtimes.

2. Check if the GPU is visible inside the container:
   ```bash
   docker exec -it sailing-assistant-gpu nvidia-smi
   ```

3. Verify environment variables in the container:
   ```bash
   docker exec -it sailing-assistant-gpu env | grep CUDA
   ```

### Memory issues

If you encounter CUDA out-of-memory errors:

1. Adjust the `PYTORCH_CUDA_ALLOC_CONF` environment variable in `docker-compose.gpu.yml`:
   ```yaml
   environment:
     - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64  # Reduce from 128 to 64
   ```

2. Restart the container:
   ```bash
   docker-compose -f docker-compose.gpu.yml down
   docker-compose -f docker-compose.gpu.yml up -d
   ```

## Performance Optimization

For optimal performance with GPU acceleration:

1. Ensure you're using the latest NVIDIA drivers compatible with CUDA 12.1
2. Consider increasing Docker's resource limits for the container
3. For multi-GPU systems, you can adjust the `CUDA_VISIBLE_DEVICES` environment variable to specify which GPU to use

## Monitoring

Monitor GPU usage with:

```bash
watch -n 1 nvidia-smi
```

This will show real-time GPU utilization, memory usage, and temperature. 