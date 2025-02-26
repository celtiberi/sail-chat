#!/bin/bash
# Script to tag and push the Docker image to Docker Hub

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
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker is not running.${NC}"
        echo "Please start the Docker daemon before running this script."
        exit 1
    fi

    echo -e "${GREEN}✓ Docker is installed and running${NC}"
}

# Check Docker Hub login
check_login() {
    print_header "Checking Docker Hub Login"
    
    # If username is already set from command line, use it
    if [ -n "$DOCKER_USERNAME" ]; then
        echo -e "${GREEN}✓ Using provided username: ${DOCKER_USERNAME}${NC}"
    else
        # Try to get Docker Hub username
        DOCKER_USERNAME=$(docker info 2>/dev/null | grep Username | cut -d: -f2 | tr -d '[:space:]')
        
        if [ -z "$DOCKER_USERNAME" ]; then
            echo -e "${YELLOW}⚠ No Docker Hub username found in Docker config${NC}"
            
            # Ask for Docker Hub username
            read -p "Enter your Docker Hub username: " DOCKER_USERNAME
            
            if [ -z "$DOCKER_USERNAME" ]; then
                echo -e "${RED}Error: Docker Hub username is required.${NC}"
                exit 1
            fi
        else
            echo -e "${GREEN}✓ Already logged in as ${DOCKER_USERNAME}${NC}"
        fi
    fi
    
    # Check if already logged in
    echo "Checking Docker Hub login status..."
    if ! docker login --username "$DOCKER_USERNAME" --password-stdin < /dev/null 2>/dev/null; then
        echo -e "${YELLOW}Please log in to Docker Hub:${NC}"
        docker login --username "$DOCKER_USERNAME"
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error: Failed to log in to Docker Hub.${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Already logged in as ${DOCKER_USERNAME}${NC}"
    fi
    
    echo -e "${GREEN}✓ Docker Hub login verified${NC}"
    return 0
}

# Build and tag the Docker image
build_and_tag() {
    print_header "Building and Tagging Docker Image"
    
    # Ask for repository name if not provided
    if [ -z "$REPO_NAME" ]; then
        read -p "Enter repository name [sailing-assistant-gpu]: " REPO_NAME
        REPO_NAME=${REPO_NAME:-sailing-assistant-gpu}
    fi
    
    # Ask for tag if not provided
    if [ -z "$TAG" ]; then
        read -p "Enter tag [latest]: " TAG
        TAG=${TAG:-latest}
    fi
    
    # Full image name
    IMAGE_NAME="$DOCKER_USERNAME/$REPO_NAME:$TAG"
    echo -e "Image will be tagged as: ${BLUE}$IMAGE_NAME${NC}"
    
    # Build the image
    echo -e "\nBuilding the Docker image..."
    docker-compose -f docker-compose.gpu.yml build
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to build the Docker image.${NC}"
        exit 1
    fi
    
    # Tag the image
    echo -e "\nTagging the image as $IMAGE_NAME..."
    docker tag sailing-assistant-gpu "$IMAGE_NAME"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to tag the Docker image.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Docker image built and tagged successfully${NC}"
    return 0
}

# Push the Docker image to Docker Hub
push_image() {
    print_header "Pushing Docker Image to Docker Hub"
    
    echo -e "Pushing ${BLUE}$IMAGE_NAME${NC} to Docker Hub..."
    docker push "$IMAGE_NAME"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to push the Docker image to Docker Hub.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Docker image pushed successfully${NC}"
    echo -e "\nYou can pull this image with:"
    echo -e "${BLUE}docker pull $IMAGE_NAME${NC}"
    return 0
}

# Main function
main() {
    print_header "Docker Hub Push Utility"
    
    # Get command line arguments
    DOCKER_USERNAME=""
    REPO_NAME=""
    TAG=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--username)
                DOCKER_USERNAME="$2"
                shift 2
                ;;
            -r|--repo)
                REPO_NAME="$2"
                shift 2
                ;;
            -t|--tag)
                TAG="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}Error: Unknown option $1${NC}"
                echo "Usage: $0 [-u|--username USERNAME] [-r|--repo REPO_NAME] [-t|--tag TAG]"
                exit 1
                ;;
        esac
    done
    
    check_docker
    check_login
    build_and_tag
    push_image
    
    print_header "Docker Hub Push Complete"
}

# Run the main function with all arguments
main "$@" 