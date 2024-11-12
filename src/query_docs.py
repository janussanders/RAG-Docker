#!/usr/bin/env python3

from typing import List, Optional
from pathlib import Path
from loguru import logger
from llama_index.core import (
    SimpleDirectoryReader, 
    Document, 
    VectorStoreIndex, 
    StorageContext,
    ServiceContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient

class DocumentQuerier:
    def __init__(
        self,
        docs_dir: str = './docs',
        collection_name: str = 'dspy_docs',
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        chunk_size: int = 1024,
        chunk_overlap: int = 200
    ):
        self.docs_dir = Path(docs_dir)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize clients
        self.qdrant_client = AsyncQdrantClient(url="qdrant", port=6333)
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model
        )
        
        # Initialize chunker
        self.chunker = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n"
        )

    async def process_documents(self) -> List[Document]:
        """Load and process documents from the docs directory."""
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

    async def query(self, query_text: str) -> str:
        """Query the vector store."""
        if not self.index:
            raise ValueError("Index not initialized. Call setup_vector_store first.")
        
        response = await self.index.aquery(query_text)
        return str(response) 