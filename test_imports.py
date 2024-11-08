#!/usr/bin/env python3

import os
import sys
import torch

# Enable Metal optimizations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("System Information:")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Using device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")

print("\nPython path:")
for path in sys.path:
    print(path)

print("\nTesting specific import:")
try:
    from llama_index_vector_stores_qdrant import QdrantVectorStore
    print("Successfully imported QdrantVectorStore")
except ImportError as e:
    print("Import failed:", e)
    print("Detailed error:", sys.exc_info()) 