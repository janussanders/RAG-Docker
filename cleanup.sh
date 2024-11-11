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

echo "Starting cleanup..."

# Clean up symlinks
echo "Cleaning up symlinks..."
find . -type l -delete 2>/dev/null || true
log_success "Symlinks cleaned"

# Clean up macOS metadata
echo "Cleaning up macOS metadata..."
find . -name "._*" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true

# Stop and remove containers
echo "Stopping containers..."
docker-compose -f docker/docker-compose.yml down || true
log_success "Containers stopped"

# Remove specific containers if they exist
for container in rag_app rag_qdrant rag_ollama; do
    if docker ps -a | grep -q $container; then
        docker rm -f $container || true
        log_success "Removed container: $container"
    fi
done

# Remove volumes
echo "Removing volumes..."
for volume in docker_ollama_models docker_qdrant_storage rag-docker_qdrant_storage; do
    if docker volume ls | grep -q $volume; then
        docker volume rm -f $volume || true
        log_success "Removed volume: $volume"
    fi
done

# Remove networks (except default ones)
echo "Removing networks..."
for network in docker_default; do
    if docker network ls | grep -q $network; then
        docker network rm $network || true
        log_success "Removed network: $network"
    fi
done

# Clean up Docker system
echo "Cleaning up Docker system..."
docker system prune -f
log_success "Docker system cleaned"

# Clean local directories
echo "Cleaning local directories..."
rm -rf qdrant_storage/* data/* logs/* cache/* 2>/dev/null || true
log_success "Local directories cleaned"

log_success "Cleanup complete!"