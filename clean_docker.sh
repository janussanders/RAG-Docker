#!/bin/bash

# Set error handling
set -e
trap 'echo "Error on line $LINENO"' ERR

echo "Cleaning Docker directory..."

# Create temp directory
mkdir -p temp_docker

# Copy main files (ignoring hidden files)
echo "Copying essential files..."
cp docker/Dockerfile temp_docker/Dockerfile
cp docker/docker-compose.yml temp_docker/docker-compose.yml
touch temp_docker/.dockerignore

# Remove old directory
echo "Removing old docker directory..."
rm -rf docker

# Create fresh directory
echo "Creating fresh docker directory..."
mkdir docker

# Move files back
echo "Restoring clean files..."
mv temp_docker/Dockerfile docker/
mv temp_docker/docker-compose.yml docker/
mv temp_docker/.dockerignore docker/

# Remove temp directory
rm -rf temp_docker

# Set permissions
echo "Setting correct permissions..."
chmod 755 docker
chmod 644 docker/Dockerfile
chmod 644 docker/docker-compose.yml
chmod 644 docker/.dockerignore

# Clear attributes
xattr -cr docker 2>/dev/null || true

echo "Verifying..."
ls -la docker/ 