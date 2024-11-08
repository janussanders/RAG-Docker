#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print success messages
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages
error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    error "Docker is not running. Please start Docker Desktop first."
fi

# Build and start the containers
echo "Building and starting containers..."
docker-compose up --build -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health >/dev/null; then
        success "Qdrant is ready"
        break
    fi
    sleep 1
done

# Show container logs
echo -e "\nContainer logs:"
docker-compose logs

success "RAG system is ready!"
echo "To view logs in real-time, run: docker-compose logs -f"
echo "To stop the system, run: docker-compose down" 