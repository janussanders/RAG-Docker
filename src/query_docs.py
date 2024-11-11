#!/usr/bin/env python3

from typing import List, Optional
from loguru import logger
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response.schema import Response
from llama_index.core.postprocessor import SentenceTransformerRerank
from qdrant_client import QdrantClient

class DocumentQuerier:
    def __init__(
        self,
        collection_name: str = "dspy_docs",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """Initialize the document querier with optimized settings."""
        try:
            # Initialize Qdrant client
            self.client = QdrantClient("localhost", port=6333)
            
            # Initialize embedding model with batching
            self.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model,
                embed_batch_size=32
            )
            
            # Setup vector store
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embed_model
            )
            
            # Initialize reranker
            self.reranker = SentenceTransformerRerank(
                model=rerank_model,
                top_n=3
            )
            
            # Create optimized service context
            service_context = ServiceContext.from_defaults(
                embed_model=self.embed_model,
                chunk_size=1024,
                chunk_overlap=200
            )
            
            # Create index with optimized settings
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                service_context=service_context
            )
            
            logger.info("DocumentQuerier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DocumentQuerier: {e}")

    def query(self, question: str, num_results: int = 3) -> Response:
        """Query the document index."""
        query_engine = self.index.as_query_engine(
            similarity_top_k=num_results,
            response_mode="no_text"  # Just return the relevant chunks
        )
        response = query_engine.query(question)
        return response

    def print_results(self, response: Response):
        """Print query results in a readable format."""
        print("\n=== Query Results ===\n")
        
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                print(f"Node: {node.text}")
        else:
            print("No source nodes found in the response.") 