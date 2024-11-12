#!/usr/bin/env python3

from typing import List, Optional
from pathlib import Path
from loguru import logger
from llama_index.core import (
    SimpleDirectoryReader, 
    Document, 
    VectorStoreIndex, 
    StorageContext,
    ServiceContext,
    Settings,
    QueryBundle
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama

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
        self.documents = None  # Add documents storage
        
        # Initialize clients
        self.qdrant_client = QdrantClient(url="qdrant", port=6333)
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model
        )
        
        self.vector_store = None
        self.index = None
        
        # Initialize LLM with a smaller model
        self.llm = Ollama(
            model="llama2:7b-chat", 
            base_url="http://host.docker.internal:11434",
            request_timeout=180.0
        )
        
        # Add this line to pull the model
        self._pull_model()

    def _pull_model(self):
        """Pull the Ollama model if it's not already available."""
        try:
            self.llm.client.pull(self.llm.model)
            logger.info(f"Successfully pulled model: {self.llm.model}")
        except Exception as e:
            logger.error(f"Failed to pull model: {self.llm.model}. Error: {str(e)}")
            raise

    async def process_documents(self) -> List[Document]:
        """Load and process documents from the docs directory."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.docs_dir}")
            
        try:
            # List all PDF files first
            pdf_files = list(self.docs_dir.glob("*.pdf"))
            logger.info(f"Found PDF files: {[f.name for f in pdf_files]}")
            
            reader = SimpleDirectoryReader(
                str(self.docs_dir),
                recursive=True,
                filename_as_id=True,
                required_exts=[".pdf"],
                num_files_limit=10
            )
            self.documents = reader.load_data()
            logger.info(f"Total documents/chunks loaded: {len(self.documents)}")
            return self.documents
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def setup_vector_store(self):
        """Initialize vector store with Qdrant."""
        try:
            if not self.documents:
                raise ValueError("No documents loaded. Call process_documents first.")
                
            # Set the default embedding model
            Settings.embed_model = self.embed_model
                
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding_function=self.embed_model
            )
            
            # Create storage context and index
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                storage_context=storage_context,
                embed_model=self.embed_model
            )
            
            logger.info("Vector store and index initialized")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

    async def query(self, query_text: str) -> str:
        """Query the vector store."""
        if not self.index:
            raise ValueError("Index not initialized. Call setup_vector_store first.")
        
        # Create query engine with our LLM
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            streaming=False
        )
        
        # Execute query
        response = query_engine.query(query_text)
        return str(response)

    def close(self):
        """Close connections."""
        if hasattr(self, 'qdrant_client'):
            self.qdrant_client.close()