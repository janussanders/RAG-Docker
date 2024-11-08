#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print success messages
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages
error() {
    echo -e "${RED}✗ $1${NC}"
    return 1
}

# Function to print warning messages
warn() {
    echo -e "${YELLOW}! $1${NC}"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    warn "Docker not found. Installing Docker..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        error "Homebrew is required to install Docker. Please install Homebrew first:"
        echo "Visit: https://brew.sh"
        exit 1
    fi
    
    # Install Docker using Homebrew
    brew install --cask docker
    
    # Wait for Docker installation
    warn "Docker installed. Please:"
    echo "1. Open Docker Desktop from your Applications folder"
    echo "2. Complete the Docker initialization"
    echo "3. Run this script again once Docker is running"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    error "Docker daemon is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl https://ollama.ai/install.sh | sh
fi

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 5  # Wait for server to start
    success "Ollama server started"
else
    success "Ollama server already running"
fi

# Pull the llama2 model if not present
if ! ollama list | grep -q "llama2"; then
    echo "Pulling llama2 model..."
    ollama pull llama2
    success "llama2 model pulled successfully"
else
    success "llama2 model already present"
fi

# Create storage directory for Qdrant if it doesn't exist
mkdir -p "$(pwd)/qdrant_storage"

# Start Qdrant
echo "Starting Qdrant..."
if ! docker ps | grep -q "qdrant"; then
    if docker ps -a | grep -q "qdrant"; then
        # Container exists but is not running
        docker start qdrant
    else
        # Create new container
        docker run -d --name qdrant \
            -p 6333:6333 \
            -p 6334:6334 \
            -v "$(pwd)/qdrant_storage:/qdrant/storage" \
            qdrant/qdrant
    fi
    
    # Wait for Qdrant to start
    echo "Waiting for Qdrant to initialize..."
    for i in {1..30}; do
        if curl -s http://localhost:6333/health &> /dev/null; then
            success "Qdrant started successfully"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            error "Qdrant failed to start within 30 seconds"
            exit 1
        fi
    done
else
    success "Qdrant already running"
fi

# Verify services
echo -e "\nVerifying services:"

# Check Ollama
if curl -s http://localhost:11434/api/tags >/dev/null; then
    success "Ollama is responding"
else
    error "Ollama is not responding"
fi

# Check Qdrant
if curl -s http://localhost:6333/health >/dev/null; then
    success "Qdrant is responding"
else
    error "Qdrant is not responding"
    exit 1
fi

echo -e "\n${GREEN}All services are ready!${NC}" 