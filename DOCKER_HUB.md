# Docker Hub Integration

This document explains how to use Docker Hub with this project, including pushing and pulling images.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed
- A [Docker Hub](https://hub.docker.com/) account

## Pushing Images to Docker Hub

### Using the Automated Script

We provide a script that automates the process of building, tagging, and pushing the Docker image to Docker Hub:

```bash
./push_to_dockerhub.sh
```

The script will:
1. Check if Docker is installed and running
2. Verify your Docker Hub login (or prompt you to log in)
3. Build the Docker image using `docker-compose.gpu.yml`
4. Tag the image with your Docker Hub username
5. Push the image to Docker Hub

You can also provide command-line arguments:

```bash
./push_to_dockerhub.sh --username your_username --repo sailing-assistant-gpu --tag v1.0
```

### Manual Process

If you prefer to do it manually:

1. **Log in to Docker Hub**:
   ```bash
   docker login
   ```

2. **Build the image**:
   ```bash
   docker-compose -f docker-compose.gpu.yml build
   ```

3. **Tag the image**:
   ```bash
   docker tag sailing-assistant-gpu your_username/sailing-assistant-gpu:latest
   ```

4. **Push the image**:
   ```bash
   docker push your_username/sailing-assistant-gpu:latest
   ```

## Pulling Images from Docker Hub

Once the image is pushed to Docker Hub, others can pull and use it:

```bash
docker pull your_username/sailing-assistant-gpu:latest
```

To run the pulled image:

```bash
docker run -p 8000:8000 --gpus all your_username/sailing-assistant-gpu:latest
```

Or create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  app:
    image: your_username/sailing-assistant-gpu:latest
    ports:
      - "8000:8000"
    volumes:
      - ./.byaldi:/app/.byaldi
      - ./data:/app/data
      - ./logs:/app/logs
      - ./chroma_db:/app/chroma_db
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Then run:

```bash
docker-compose up -d
```

## Creating a Public Repository

By default, Docker Hub repositories are public. If you want to create a private repository, you'll need a paid Docker Hub account.

To create a public repository:

1. Log in to [Docker Hub](https://hub.docker.com/)
2. Click on "Create Repository"
3. Enter "sailing-assistant-gpu" as the name
4. Set visibility to "Public"
5. Click "Create"

## Automated Builds

Docker Hub supports automated builds from GitHub or Bitbucket repositories. To set up automated builds:

1. Log in to [Docker Hub](https://hub.docker.com/)
2. Go to your repository
3. Click on "Builds"
4. Connect your GitHub or Bitbucket account
5. Select the repository
6. Configure build rules
7. Click "Save and Build"

## Best Practices

1. **Use specific tags**: Instead of always using `latest`, use version numbers or descriptive tags like `v1.0`, `gpu-cuda12.1`, etc.

2. **Document your images**: Add a description and README to your Docker Hub repository.

3. **Scan for vulnerabilities**: Docker Hub provides vulnerability scanning for images.

4. **Use multi-stage builds**: This reduces the final image size.

5. **Keep credentials secure**: Never hardcode credentials in your Dockerfile or images.

## Troubleshooting

### Push Access Denied

If you get "access denied" when pushing:

```
denied: requested access to the resource is denied
```

Make sure:
- You're logged in to Docker Hub (`docker login`)
- You have permission to push to the repository
- The repository exists on Docker Hub

### Image Too Large

If your image is too large to push efficiently:

1. Use multi-stage builds
2. Remove unnecessary files
3. Use `.dockerignore` to exclude files
4. Consider using Docker Hub's Large Image Support

### Rate Limiting

Docker Hub has rate limits for pulls. If you hit these limits:

1. Authenticate pulls (even for public images)
2. Consider Docker Hub's paid plans
3. Set up a local registry mirror 