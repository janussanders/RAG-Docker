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

# Function to print warning messages
warn() {
    echo -e "${YELLOW}! $1${NC}"
}

PROJECT_DIR="/Volumes/Algernon/RAG/RAG-Docker"

echo "Setting permissions for RAG project..."

# Set permissions for core files
echo "Setting core file permissions..."
chmod 644 "$PROJECT_DIR/src/"*.py
chmod 644 "$PROJECT_DIR/tests/"*.py
chmod 644 "$PROJECT_DIR/docker/Dockerfile"
chmod 644 "$PROJECT_DIR/docker/docker-compose.yml"
chmod 644 "$PROJECT_DIR/requirements.txt"
success "Core file permissions set"

# Set permissions for scripts
echo "Setting script permissions..."
chmod 755 "$PROJECT_DIR/"*.sh