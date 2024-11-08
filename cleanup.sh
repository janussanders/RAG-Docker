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

echo "Starting comprehensive cleanup..."

# Stop running containers and remove Docker artifacts
echo "Cleaning Docker artifacts..."
docker-compose down 2>/dev/null
docker container prune -f
docker volume prune -f
docker image prune -f
success "Cleaned Docker artifacts"

# Deactivate virtual environment if active
deactivate 2>/dev/null || true

# Remove virtual environment and its symlinks
echo "Removing virtual environment..."
rm -rf "$PROJECT_DIR/rag_env"
rm -f "$PROJECT_DIR/_rag_env"  # Remove specific symlink
rm -f "$PROJECT_DIR/._rag_env"  # Remove potential hidden symlink
success "Removed virtual environment and related symlinks"

# Find and remove ALL symlinks recursively
echo "Removing all symlinks..."
find "$PROJECT_DIR" -type l -exec rm -f {} \;
# Double-check for any remaining symlinks with different methods
find "$PROJECT_DIR" -lname '*' -delete
find "$PROJECT_DIR" -xtype l -delete
success "Removed all symlinks"

# Remove extended attributes and ._ files
echo "Removing extended attributes..."
find "$PROJECT_DIR" -name "._*" -delete
xattr -cr "$PROJECT_DIR" 2>/dev/null
success "Removed extended attributes"

# Clean Python cache files
echo "Cleaning Python cache..."
find "$PROJECT_DIR" -type f -name "*.pyc" -delete
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find "$PROJECT_DIR" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
success "Cleaned Python cache"

# Remove cache directories
echo "Removing cache directories..."
rm -rf "$PROJECT_DIR/cache"
success "Removed cache directories"

# Verify no symlinks remain
remaining_symlinks=$(find "$PROJECT_DIR" -type l)
if [ -n "$remaining_symlinks" ]; then
    warn "Found remaining symlinks:"
    echo "$remaining_symlinks"
    echo "Attempting forced removal..."
    find "$PROJECT_DIR" -type l -exec rm -f {} \;
    success "Forced removal complete"
else
    success "No remaining symlinks found"
fi

# Clean Qdrant storage but preserve directory
echo "Cleaning Qdrant storage..."
rm -rf "$PROJECT_DIR/qdrant_storage/*"
mkdir -p "$PROJECT_DIR/qdrant_storage"
chmod 755 "$PROJECT_DIR/qdrant_storage"
success "Cleaned Qdrant storage"

echo -e "\n${GREEN}Cleanup complete!${NC}"
echo "You can now run:"
echo "1. ./setup.sh to create a new environment"
echo "2. ./set_permissions.sh to set correct permissions"
echo "3. source ./activate_rag.sh to activate the environment"