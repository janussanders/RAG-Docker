#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print success messages
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Set project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_COMPOSE="docker/docker-compose.yml"

# Add warning prompt
echo -e "${RED}⚠️  WARNING: This will erase all Docker containers, volumes, and cached data${NC}"
echo "Are you sure you want to proceed? (y/N) "
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo "Starting comprehensive cleanup..."

# Clean macOS metadata and extended attributes
echo "Cleaning up macOS metadata and symlinks..."
find "$PROJECT_DIR" -name "._*" -delete && xattr -cr "$PROJECT_DIR"
success "Cleaned up macOS metadata"

# Clean Docker artifacts
echo "Cleaning Docker artifacts..."
docker-compose -f "$DOCKER_COMPOSE" down --volumes --remove-orphans 2>/dev/null
docker container prune -f
docker volume prune -f
docker image prune -f
docker builder prune -f
success "Cleaned Docker artifacts"

# Clean Qdrant storage
echo "Cleaning Qdrant storage..."
rm -rf "$PROJECT_DIR/qdrant_storage"
rm -rf "$PROJECT_DIR/._qdrant_storage"
mkdir -p "$PROJECT_DIR/qdrant_storage"
chmod 755 "$PROJECT_DIR/qdrant_storage"
xattr -cr "$PROJECT_DIR/qdrant_storage" 2>/dev/null || true
success "Cleaned Qdrant storage"

# Clean build context
echo "Cleaning build context..."
rm -rf "$PROJECT_DIR/qdrant_storage/aliases" 2>/dev/null
rm -rf "$PROJECT_DIR/qdrant_storage/collections" 2>/dev/null
rm -rf "$PROJECT_DIR/qdrant_storage/snapshots" 2>/dev/null
success "Cleaned build context"

echo -e "\n${GREEN}Cleanup complete!${NC}"
echo "You can now run:"
echo "1. ./setup.sh to create a new environment" 