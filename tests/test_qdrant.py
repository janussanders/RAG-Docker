#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import UpdateStatus
import pytest
from src.query_docs import DocumentQuerier

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

def test_qdrant_connection():
    client = QdrantClient("localhost", port=6333)
    assert client.get_collections()

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
    if test_qdrant_connection():
        success("\nAll Qdrant tests passed successfully!")
        sys.exit(0)
    else:
        error("\nQdrant tests failed. Please check the logs for more details.")
        sys.exit(1) 