#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import UpdateStatus

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
NC = '\033[0m'  # No Color

def success(msg):
    print(f"{GREEN}✓ {msg}{NC}")

def error(msg):
    print(f"{RED}✗ {msg}{NC}")

def warn(msg):
    print(f"{YELLOW}! {msg}{NC}")

def test_qdrant():
    print("\nTesting Qdrant Connection and Functionality...")
    
    try:
        # Initialize client
        client = QdrantClient("localhost", port=6333)
        success("Connected to Qdrant")

        # List existing collections
        collections = client.get_collections()
        success(f"Retrieved collections: {collections}")

        # Create a test collection
        test_collection_name = f"test_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        client.create_collection(
            collection_name=test_collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        success(f"Created test collection: {test_collection_name}")

        # Wait a moment for collection to be ready
        time.sleep(1)

        # Verify collection was created
        collections_after = client.get_collections()
        if any(c.name == test_collection_name for c in collections_after.collections):
            success("Test collection verified")
        else:
            error("Test collection not found after creation")

        # Clean up test collection
        client.delete_collection(test_collection_name)
        success("Cleaned up test collection")

        return True

    except Exception as e:
        error(f"Error during Qdrant test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting Qdrant Test Suite")
    print("=========================")
    
    # Check if Qdrant service is running via Docker
    import subprocess
    try:
        docker_ps = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        if docker_ps.stdout.strip():
            success("Qdrant container is running")
        else:
            warn("Qdrant container not found. Starting container...")
            subprocess.run(["docker-compose", "up", "-d", "qdrant"])
            success("Started Qdrant container")
    except Exception as e:
        error(f"Error checking Docker status: {str(e)}")
        sys.exit(1)

    # Run tests
    if test_qdrant():
        success("\nAll Qdrant tests passed successfully!")
        sys.exit(0)
    else:
        error("\nQdrant tests failed. Please check the logs for more details.")
        sys.exit(1) 