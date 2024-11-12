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

echo "Starting RAG services..."

# Verify clean state
if [ -f "docker/._Dockerfile" ] || [ -f "docker/._docker-compose.yml" ] || [ -f "._docker" ]; then
    log_warning "Environment not clean. Would you like to run cleanup.sh? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        ./cleanup.sh
    else
        log_error "Exiting due to unclean environment"
        exit 1
    fi
fi

# Clean up macOS metadata files
echo "Cleaning up macOS metadata..."
find . -name "._*" -delete
find . -name ".DS_Store" -delete
log_success "Metadata cleanup complete"

# Start services with docker-compose
echo "Building and starting containers..."
docker-compose -f docker/docker-compose.yml up -d --build

# Verify services started
echo "Verifying services..."
docker-compose -f docker/docker-compose.yml ps

log_success "Services started successfully!"

# Ask about running setup
log_warning "Would you like to run setup.sh? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    ./setup.sh
fi