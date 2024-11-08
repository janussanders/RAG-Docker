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
chmod 644 "$PROJECT_DIR/requirements.txt"
chmod 644 "$PROJECT_DIR/Dockerfile"
chmod 644 "$PROJECT_DIR/docker-compose.yml"
chmod 644 "$PROJECT_DIR/main.py"
success "Core file permissions set"

# Set permissions for scripts
echo "Setting script permissions..."
chmod 755 "$PROJECT_DIR/setup.sh"
chmod 755 "$PROJECT_DIR/cleanup.sh"
chmod 755 "$PROJECT_DIR/activate_rag.sh"
chmod 755 "$PROJECT_DIR/start_services.sh"
chmod 755 "$PROJECT_DIR/set_permissions.sh" 