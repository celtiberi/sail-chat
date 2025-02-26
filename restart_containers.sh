#!/bin/bash

# Script to restart Docker containers for the sailing assistant application
# This helps users apply changes to their environment

# Print colored output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Sailing Assistant Docker Restart Utility ===${NC}"
echo -e "${YELLOW}This script will help you restart your Docker containers${NC}"
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${YELLOW}Docker is not running. Please start Docker first.${NC}"
  exit 1
fi

# Detect system type
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == arm* ]]; then
  IS_APPLE_SILICON=true
  echo -e "${BLUE}Detected Apple Silicon Mac${NC}"
else
  IS_APPLE_SILICON=false
  
  # Check for NVIDIA GPU
  if command -v nvidia-smi > /dev/null 2>&1; then
    HAS_NVIDIA_GPU=true
    echo -e "${BLUE}Detected NVIDIA GPU${NC}"
  else
    HAS_NVIDIA_GPU=false
    echo -e "${BLUE}No NVIDIA GPU detected${NC}"
  fi
fi

# Ask which configuration to use
echo
echo -e "${YELLOW}Which configuration would you like to use?${NC}"
echo "1) CPU Mode (recommended for Apple Silicon and systems without GPU)"
echo "2) GPU Mode (requires NVIDIA GPU)"
read -p "Enter your choice (1/2): " CHOICE

# Set the docker-compose file based on choice
if [ "$CHOICE" == "2" ]; then
  COMPOSE_FILE="docker-compose.gpu.yml"
  echo -e "${BLUE}Using GPU configuration${NC}"
else
  COMPOSE_FILE="docker-compose.yml"
  echo -e "${BLUE}Using CPU configuration${NC}"
fi

# Stop any running containers
echo
echo -e "${YELLOW}Stopping any running containers...${NC}"
docker-compose -f $COMPOSE_FILE down

# Remove old containers and images
echo
echo -e "${YELLOW}Removing old containers...${NC}"
docker-compose -f $COMPOSE_FILE rm -f

# Rebuild and start containers
echo
echo -e "${YELLOW}Rebuilding and starting containers...${NC}"
docker-compose -f $COMPOSE_FILE up -d --build

# Check if containers are running
echo
echo -e "${YELLOW}Checking container status...${NC}"
docker-compose -f $COMPOSE_FILE ps

echo
echo -e "${GREEN}Restart complete!${NC}"
echo -e "${BLUE}The application should be available at http://localhost:8000${NC}"
echo
echo -e "${YELLOW}To view logs:${NC} docker-compose -f $COMPOSE_FILE logs -f" 