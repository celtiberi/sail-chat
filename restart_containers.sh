#!/bin/bash

# Script to restart containers with the appropriate configuration

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "\n${BLUE}=========================================${NC}"
    echo -e "${BLUE}   $1${NC}"
    echo -e "${BLUE}=========================================${NC}\n"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed.${NC}"
        echo "Please install Docker before running this script."
        echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker is not running.${NC}"
        echo "Please start the Docker daemon before running this script."
        exit 1
    fi

    echo -e "${GREEN}✓ Docker is installed and running${NC}"
}

# Detect system type
detect_system() {
    print_header "Detecting System"
    
    # Check if running on macOS
    if [[ "$(uname)" == "Darwin" ]]; then
        # Check if running on Apple Silicon
        if [[ "$(uname -m)" == "arm64" ]]; then
            echo -e "${GREEN}✓ Detected Apple Silicon Mac${NC}"
            SYSTEM_TYPE="apple_silicon"
        else
            echo -e "${GREEN}✓ Detected Intel Mac${NC}"
            SYSTEM_TYPE="cpu"
        fi
    # Check if running on Linux
    elif [[ "$(uname)" == "Linux" ]]; then
        echo -e "${GREEN}✓ Detected Linux${NC}"
        
        # Check for NVIDIA GPU
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
            echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
            
            # Check for NVIDIA Docker runtime
            if docker info | grep -q "nvidia"; then
                echo -e "${GREEN}✓ NVIDIA Docker runtime detected${NC}"
                SYSTEM_TYPE="gpu"
            else
                echo -e "${YELLOW}⚠ NVIDIA GPU detected but NVIDIA Docker runtime not found${NC}"
                echo "Please install the NVIDIA Container Toolkit:"
                echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
                SYSTEM_TYPE="cpu"
            fi
        else
            echo -e "${YELLOW}⚠ No NVIDIA GPU detected${NC}"
            SYSTEM_TYPE="cpu"
        fi
    else
        echo -e "${YELLOW}⚠ Unknown system: $(uname)${NC}"
        SYSTEM_TYPE="cpu"
    fi
    
    echo -e "\nSystem type: ${BLUE}${SYSTEM_TYPE}${NC}"
}

# Stop and remove existing containers
stop_containers() {
    print_header "Stopping Containers"
    
    # Check if containers are running
    if docker ps -q --filter "name=sailing-assistant" &> /dev/null || docker ps -q --filter "name=sailing-assistant-gpu" &> /dev/null; then
        echo "Stopping and removing existing containers..."
        docker-compose down &> /dev/null
        docker-compose -f docker-compose.gpu.yml down &> /dev/null
        echo -e "${GREEN}✓ Containers stopped and removed${NC}"
    else
        echo -e "${YELLOW}⚠ No running containers found${NC}"
    fi
}

# Start containers with the appropriate configuration
start_containers() {
    print_header "Starting Containers"
    
    if [[ "$SYSTEM_TYPE" == "gpu" ]]; then
        echo "Starting containers with GPU configuration..."
        docker-compose -f docker-compose.gpu.yml up -d
        CONTAINER_NAME="sailing-assistant-gpu"
    else
        echo "Starting containers with CPU configuration..."
        docker-compose up -d
        CONTAINER_NAME="sailing-assistant"
    fi
    
    # Check if container started successfully
    if docker ps -q --filter "name=$CONTAINER_NAME" &> /dev/null; then
        echo -e "${GREEN}✓ Container started successfully${NC}"
        echo -e "\nContainer logs (press Ctrl+C to exit):"
        docker logs -f "$CONTAINER_NAME"
    else
        echo -e "${RED}✗ Failed to start container${NC}"
        echo "Check the logs for more information:"
        echo "docker logs $CONTAINER_NAME"
    fi
}

# Main function
main() {
    print_header "Sailing Assistant Container Manager"
    
    check_docker
    detect_system
    stop_containers
    start_containers
}

# Run the main function
main 