#!/bin/bash
# Simple script to build, tag, and push the Docker image to Docker Hub

# Set variables
USERNAME="pwcremin"
REPO_NAME="sailing-assistant-gpu"
TAG="latest"

# Build the image
echo "Building the Docker image..."
docker-compose -f docker-compose.gpu.yml build

# Tag the image
echo "Tagging the image as $USERNAME/$REPO_NAME:$TAG..."
docker tag sailing-assistant-gpu "$USERNAME/$REPO_NAME:$TAG"

# Push the image
echo "Pushing $USERNAME/$REPO_NAME:$TAG to Docker Hub..."
docker push "$USERNAME/$REPO_NAME:$TAG"

echo "Done! You can pull this image with:"
echo "docker pull $USERNAME/$REPO_NAME:$TAG" 