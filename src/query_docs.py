from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.response.schema import Response
from qdrant_client import QdrantClient

class DocumentQuerier:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient("localhost", port=6333)
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Setup vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name="dspy_docs",
            embedding_dimension=384
        )
        
        # Create index
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)

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