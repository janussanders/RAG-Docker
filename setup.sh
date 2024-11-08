#!/bin/bash

# Define project directory and paths
PROJECT_DIR="/Volumes/Algernon/RAG/RAG-Docker"
DOCKER_COMPOSE="$PROJECT_DIR/docker/docker-compose.yml"

# Import color and message functions
source ./utils.sh

echo "Starting RAG system setup..."

# 1. Set permissions first
if ! ./set_permissions.sh; then
    error "Failed to set permissions"
    exit 1
fi

# 2. Clean up any existing containers and volumes
if ! ./cleanup.sh; then
    error "Failed to clean up existing containers"
    exit 1
fi

# 3. Start basic services (Ollama)
if ! ./start_services.sh; then
    error "Failed to start basic services"
    exit 1
fi

# 4. Start Qdrant separately
if ! ./start_qdrant.sh; then
    error "Failed to start Qdrant"
    exit 1
fi

# Pre-build cleanup
echo "Preparing build environment..."
find "$PROJECT_DIR" -name "._*" -exec rm -f {} \;
xattr -cr "$PROJECT_DIR" 2>/dev/null || true

# 5. Build and start RAG container
echo "Building and starting RAG container..."
if ! DOCKER_BUILDKIT=1 docker-compose -f "$DOCKER_COMPOSE" build; then
    error "Failed to build RAG container"
    exit 1
fi

if ! docker-compose -f "$DOCKER_COMPOSE" up -d rag; then
    error "Failed to start RAG container"
    exit 1
fi

# 6. Install additional requirements if needed
if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements..."
    pip install -r requirements.txt
    success "Installed additional requirements"
fi

# 7. Verify all services
echo "Verifying all services..."
if ! curl -s -f "http://localhost:11434/api/tags" > /dev/null; then
    error "Ollama is not responding"
    exit 1
fi

if ! curl -s -f "http://localhost:6333/collections" > /dev/null; then
    error "Qdrant is not responding"
    exit 1
fi

success "Setup complete!"
echo "To activate the environment, use: source ./activate_rag.sh"


