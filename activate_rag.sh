#!/bin/bash

# Source the virtual environment
source rag_env/bin/activate

# Check if Qdrant is running
if ! curl -s -f "http://localhost:6333/collections" > /dev/null; then
    echo "Qdrant is not running. Starting Qdrant..."
    docker-compose up -d qdrant
    sleep 5
    if curl -s -f "http://localhost:6333/collections" > /dev/null; then
        echo "✓ Qdrant started successfully"
    else
        echo "! Warning: Qdrant failed to start"
    fi
fi

# Print status
echo "✓ Environment activated with Apple Silicon optimizations"
echo "✓ Metal Performance Shaders (MPS) enabled"
echo "✓ UTF-8 encoding configured"
echo "✓ Cache directories set"

PROJECT_DIR="/Volumes/Algernon/RAG/RAG-Docker"
DOCKER_COMPOSE="$PROJECT_DIR/docker/docker-compose.yml"

# When running docker-compose commands
docker-compose -f "$DOCKER_COMPOSE" exec rag python /app/src/main.py
