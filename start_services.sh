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
    log_error "Environment not clean. Please run ./cleanup.sh first"
    exit 1
fi

# Start services with docker-compose
echo "Building and starting containers..."
docker-compose -f docker/docker-compose.yml up -d --build

# Verify services started
echo "Verifying services..."
docker-compose -f docker/docker-compose.yml ps

log_success "Services started successfully!"