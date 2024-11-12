#!/usr/bin/env python3

import os
import asyncio
from typing import List, Optional
from pathlib import Path
from loguru import logger
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from llama_index.llms.ollama import Ollama

class DocumentProcessor:
    def __init__(
        self, 
        docs_dir: str = './docs', 
        collection_name: str = 'dspy_docs',
        embedding_model: str = 'BAAI/bge-small-en-v1.5'
    ):
        self.docs_dir = Path(docs_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.qdrant_client = AsyncQdrantClient(
            url="qdrant",
            port=6333,
            timeout=60.0
        )
        self.vector_store = None
        self.index = None
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model
        )
        
    async def initialize(self):
        """Async initialization of the processor."""
        await self._init_collection()
        self._init_vector_store()
        logger.info(f"Initialized collection: {self.collection_name}")
    
    async def _init_collection(self, vector_size: int = 384):
        """Initialize Qdrant collection if it doesn't exist."""
        collections = await self.qdrant_client.get_collections()
        if not any(c.name == self.collection_name for c in collections.collections):
            await self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _init_vector_store(self):
        """Initialize the vector store with Qdrant."""
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding_function=self.embed_model
        )
        logger.info("Initialized vector store")
    
    async def load_documents(self) -> List[str]:
        """Load PDF documents from the specified directory."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.docs_dir}")
            
        try:
            reader = SimpleDirectoryReader(
                str(self.docs_dir),
                recursive=True,
                filename_as_id=True,
                file_extractor={
                    ".pdf": "default"
                }
            )
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents from {self.docs_dir}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    async def get_collection_info(self):
        """Get information about the current collection."""
        return await self.qdrant_client.get_collection(self.collection_name)

    async def close(self):
        """Close the Qdrant client connection."""
        await self.qdrant_client.close()