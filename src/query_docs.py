#!/usr/bin/env python3

from typing import List, Optional
from pathlib import Path
from loguru import logger
from llama_index.core import (
    SimpleDirectoryReader, 
    Document, 
    VectorStoreIndex, 
    StorageContext,
    Settings,
    QueryBundle,
    PromptTemplate,
    ServiceContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
import uuid  # Add this import at the top
import asyncio
import json
import sys
import torch
import os

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    "app.log",  # Log to this file
    rotation="500 MB",  # Rotate when file reaches 500MB
    level="DEBUG",  # Set minimum level
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    enqueue=True  # Thread-safe logging
)
# Also add console output
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    enqueue=True
)

# Define a system message to set context for the model
SYSTEM_MESSAGE = """You are a helpful AI assistant that answers questions based on the provided context. 
If you don't know the answer or can't find it in the context, simply say "I don't know!"."""

# Define a single, clear QA template
QA_TEMPLATE = ("""You are a helpful AI assistant. Use the following context to answer the question. 
If you cannot find the answer in the context, say "I cannot find the answer in the provided context."

Context:
{context_str}

Question: {query_str}

Answer: """)

# Create a single prompt template instance
qa_prompt_tmpl = PromptTemplate(
    template=QA_TEMPLATE,
    system_message=SYSTEM_MESSAGE
)

class DocumentQuerier:
    def __init__(self, docs_dir: Path, collection_name: str):
        self.docs_dir = docs_dir
        self.collection_name = collection_name
        self.query_engine = None
        
        # Initialize immediately
        self.setup_vector_store()
        self.initialize_query_engine()
        
    def setup_vector_store(self):
        """Setup the vector store"""
        try:
            logger.info("Setting up vector store...")
            
            # Initialize embedding model
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            
            # Initialize Qdrant client with explicit timeout
            client = QdrantClient(
                "qdrant", 
                port=6333,
                timeout=60.0  # Increase timeout
            )
            
            # Ensure collection exists
            client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "size": 384,  # BGE-small dimension
                    "distance": "Cosine"
                }
            )
            
            # Setup vector store
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name
            )
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load documents
            documents = SimpleDirectoryReader(self.docs_dir).load_data()
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                embed_model=embed_model
            )
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise
            
    def initialize_query_engine(self):
        """Initialize the query engine"""
        try:
            logger.info("Initializing query engine...")
            
            # Initialize LLM with robust connection settings
            llm = Ollama(
                model="orca-mini",  # Make sure this matches your pulled model
                base_url="http://ollama:11434",
                request_timeout=120.0,  # Increase timeout
                additional_kwargs={
                    "num_ctx": 4096,
                    "num_predict": 256,
                }
            )
            
            # Test connection
            try:
                logger.info("Testing Ollama connection...")
                llm.complete("test")
                logger.info("Ollama connection successful!")
            except Exception as e:
                logger.error(f"Ollama connection test failed: {str(e)}")
                raise
            
            # Rest of the initialization...
            rerank = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=3
            )
            
            self.query_engine = self.index.as_query_engine(
                llm=llm,
                node_postprocessors=[rerank],
                text_qa_template=qa_prompt_tmpl
            )
            
        except Exception as e:
            logger.error(f"Error initializing query engine: {str(e)}")
            raise
        
    def query(self, query_text: str) -> str:
        """Query the documentation"""
        try:
            if self.query_engine is None:
                raise ValueError("Query engine not initialized")
                
            response = self.query_engine.query(query_text)
            return response.response
            
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return f"Error processing query: {str(e)}"