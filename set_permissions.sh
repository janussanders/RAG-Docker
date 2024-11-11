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

# Set permissions for Docker files
echo "Setting Docker file permissions..."

# Aggressive cleanup of Docker directory
echo "Deep cleaning Docker directory..."
rm -f "$PROJECT_DIR/docker/._Dockerfile" "$PROJECT_DIR/docker/._docker-compose.yml"
find "$PROJECT_DIR/docker" -name "._*" -delete
find "$PROJECT_DIR/docker" -type f -exec xattr -c {} \; 2>/dev/null || true
xattr -cr "$PROJECT_DIR/docker" 2>/dev/null || true

# Reset Docker directory permissions
chmod 755 "$PROJECT_DIR/docker"
success "Docker directory reset"

# Handle Dockerfile specifically
if [ -f "$PROJECT_DIR/docker/Dockerfile" ]; then
    rm -f "$PROJECT_DIR/docker/._Dockerfile"
    xattr -c "$PROJECT_DIR/docker/Dockerfile" 2>/dev/null || true
    chmod 644 "$PROJECT_DIR/docker/Dockerfile"
    success "Dockerfile permissions reset"
else
    error "Dockerfile not found!"
    exit 1
fi

# Handle docker-compose.yml specifically
if [ -f "$PROJECT_DIR/docker/docker-compose.yml" ]; then
    rm -f "$PROJECT_DIR/docker/._docker-compose.yml"
    xattr -c "$PROJECT_DIR/docker/docker-compose.yml" 2>/dev/null || true
    chmod 644 "$PROJECT_DIR/docker/docker-compose.yml"
    success "docker-compose.yml permissions reset"
fi

# Verify permissions
if [ ! -r "$PROJECT_DIR/docker/Dockerfile" ]; then
    error "Dockerfile not readable after permission setup!"
    exit 1
fi

success "Docker files configured"

clean_docker_directory() {
    echo "Deep cleaning Docker directory..."
    
    # Create a completely new temp directory outside the project
    TEMP_DIR="/tmp/docker_clean_$$"
    mkdir -p "$TEMP_DIR"
    
    echo "Copying essential files to temp directory..."
    # Copy only specific files, ignoring all hidden files
    cat docker/Dockerfile > "$TEMP_DIR/Dockerfile"
    cat docker/docker-compose.yml > "$TEMP_DIR/docker-compose.yml"
    echo "" > "$TEMP_DIR/.dockerignore"
    
    echo "Removing old docker directory..."
    rm -rf docker
    
    echo "Creating fresh docker directory..."
    mkdir -p docker
    
    echo "Copying clean files back..."
    cp "$TEMP_DIR/Dockerfile" docker/
    cp "$TEMP_DIR/docker-compose.yml" docker/
    cp "$TEMP_DIR/.dockerignore" docker/
    
    echo "Setting ownership and permissions..."
    # Set ownership and permissions
    chown -R $(id -u):$(id -g) docker
    chmod 755 docker
}

main() {
    echo "Setting permissions for containerized RAG system..."
    
    # Create necessary directories
    check_directory "$PROJECT_DIR/src"
    check_directory "$PROJECT_DIR/tests"
    check_directory "$PROJECT_DIR/docker"
    check_directory "$DATA_DIR"
    check_directory "$QDRANT_STORAGE"
    
    # Set permissions
    set_existing_permissions
    set_core_permissions
    set_docker_permissions  # This now includes the deep clean
    set_data_permissions
    set_script_permissions
    set_container_volume_permissions
    
    # Verify all permissions
    verify_permissions
    
    log_success "All permissions set successfully!"
    # ... rest of the main function ...
}