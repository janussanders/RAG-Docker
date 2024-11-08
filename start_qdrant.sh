#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print success messages
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages
error() {
    echo -e "${RED}✗ $1${NC}"
}

echo "Starting Qdrant setup..."

# Stop any running containers and remove volumes
echo "Cleaning up existing containers and volumes..."
docker-compose down -v

# Start Qdrant
echo "Starting Qdrant container..."
docker-compose up -d

# Wait for container to be ready
echo "Waiting for Qdrant to initialize..."
sleep 10

# Check container status
CONTAINER_ID=$(docker ps -q --filter name=qdrant)
if [ -z "$CONTAINER_ID" ]; then
    error "Container failed to start"
    exit 1
fi

# Show container logs
echo "Container logs:"
docker logs $CONTAINER_ID

# Check if Qdrant is responding
echo "Testing Qdrant connection..."
if curl -s -f "http://localhost:6333/collections" > /dev/null; then
    success "Qdrant is running and responding"
else
    error "Qdrant failed to start properly"
    docker logs $CONTAINER_ID
    exit 1
fi

success "Qdrant setup complete!"