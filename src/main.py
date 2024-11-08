#!/usr/bin/env python3
# Script to Set Up Qdrant Vector Database for RAG System

import os
import torch
import time

# Enable Metal optimizations for Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Metal Performance Shaders (MPS) for acceleration")
else:
    device = torch.device("cpu")
    print("! Using CPU (MPS not available)")

import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index_llms_ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank

def wait_for_services():
    """Wait for Qdrant and Ollama to be ready"""
    import requests
    from requests.exceptions import RequestException
    
    services = {
        'Qdrant': 'http://localhost:6333/health',
        'Ollama': 'http://localhost:11434/api/tags'
    }
    
    for service, url in services.items():
        print(f"Waiting for {service}...")
        for _ in range(30):  # Try for 30 seconds
            try:
                requests.get(url)
                print(f"✓ {service} is ready")
                break
            except RequestException:
                time.sleep(1)
        else:
            raise RuntimeError(f"{service} is not available")

def main():
    # Wait for services to be ready
    wait_for_services()
    
    print("Setting up Qdrant vector store...")
    
    # Initialize Qdrant client with actual server
    client = qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )
    
    # Create vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_documents"
    )
    print("✓ Vector store initialized successfully")
    
    # Initialize LLM
    llm = Ollama(model="llama3.2", temperature=0.1)
    print("✓ LLM initialized successfully")
    
    # Initialize embedding model with optimizations
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )
    print("✓ Embedding model initialized successfully")
    
    # Create service context
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    print("✓ Service context created successfully")

if __name__ == "__main__":
    main()
