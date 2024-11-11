#!/bin/bash

# Set error handling
set -e
trap 'echo "Error on line $LINENO"' ERR

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Helper functions
log_success() { echo -e "${GREEN}✓ $1${NC}"; }
log_warning() { echo -e "${YELLOW}! $1${NC}"; }
log_error() { echo -e "${RED}✗ $1${NC}" >&2; }

# Configuration
MAX_RETRIES=30
RETRY_DELAY=2

echo "Starting RAG services..."

# Ensure permissions are correct
./set_permissions.sh

# Start services with docker-compose
echo "Building and starting containers..."
docker-compose -f docker/docker-compose.yml up -d --build
log_success "Containers started"

# Wait for Qdrant
echo "Waiting for Qdrant to be ready..."
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s -f "http://localhost:6333/health" > /dev/null; then
        log_success "Qdrant is ready"
        break
    fi
    if [ $i -eq $MAX_RETRIES ]; then
        log_error "Qdrant failed to start"
        exit 1
    fi
    echo "Waiting for Qdrant... ($i/$MAX_RETRIES)"
    sleep $RETRY_DELAY
done

# Wait for Ollama
echo "Waiting for Ollama to be ready..."
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s -f "http://localhost:11434/api/tags" > /dev/null; then
        log_success "Ollama is ready"
        break
    fi
    if [ $i -eq $MAX_RETRIES ]; then
        log_error "Ollama failed to start"
        exit 1
    fi
    echo "Waiting for Ollama... ($i/$MAX_RETRIES)"
    sleep $RETRY_DELAY
done

# Verify all services
echo "Verifying services..."
docker-compose -f docker/docker-compose.yml ps

log_success "All services started successfully!"
echo -e "\nUseful commands:"
echo "- View logs: docker-compose -f docker/docker-compose.yml logs"
echo "- Stop services: docker-compose -f docker/docker-compose.yml down"
echo "- Restart services: docker-compose -f docker/docker-compose.yml restart"
echo "- Check logs: docker-compose -f docker/docker-compose.yml logs"
echo "- Check status: docker-compose -f docker/docker-compose.yml ps"