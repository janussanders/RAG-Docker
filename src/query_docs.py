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
    QueryBundle,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank

# Define a system message to set context for the model
SYSTEM_MESSAGE = """You are a helpful AI assistant that answers questions based on the provided context. 
If you don't know the answer or can't find it in the context, simply say "I don't know!"."""

# Define the prompt template with proper formatting
template = """Context information is below:
---------------------
{context_str}
---------------------

Please think step by step to answer the following query in a crisp manner.
If you can't find the answer in the context, say "I don't know!".

Query: {query_str}

Answer: Let me help you with that."""

qa_prompt_tmpl = PromptTemplate(
    template=template,
    system_message=SYSTEM_MESSAGE
)

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
        
        # Initialize LLM with the smaller quantized model
        logger.info("Initializing Ollama with orca-mini model...")
        self.llm = Ollama(
            model="orca-mini",
            base_url="http://host.docker.internal:11434",
            request_timeout=180.0
        )
        
        # Add this line to pull the model
        logger.info("Pulling Ollama model...")
        self._pull_model()
        
        # Verify model is loaded by making a test query
        self._verify_model()
        
        # Initialize reranker
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=3
        )
        
        self.query_engine = None  # Will be initialized after index creation

    def _pull_model(self):
        """Pull the Ollama model if it's not already available."""
        try:
            logger.info(f"Attempting to pull model: {self.llm.model}")
            self.llm.client.pull(self.llm.model)
            logger.info(f"Successfully pulled model: {self.llm.model}")
        except Exception as e:
            logger.error(f"Failed to pull model: {self.llm.model}. Error: {str(e)}")
            raise

    def _verify_model(self):
        """Verify the model is loaded by making a test query."""
        try:
            logger.info("Testing model with simple query...")
            test_response = self.llm.complete("Say 'hello'")
            logger.info(f"Model test successful. Response: {test_response}")
        except Exception as e:
            logger.error(f"Model verification failed. Error: {str(e)}")
            raise RuntimeError("Failed to verify Ollama model is working properly")

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
            # Set the default LLM to our Ollama instance
            Settings.llm = self.llm
                
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
            
            # Initialize query engine with custom settings
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=10,
                node_postprocessors=[self.reranker],
                llm=self.llm,
                text_qa_template=qa_prompt_tmpl
            )
            
            logger.info("Vector store and index initialized")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

    async def query(self, question: str) -> str:
        try:
            # Get initial documents from vector store
            initial_results = self.vector_store.similarity_search(question, k=5)  # Get more initial results
            
            # Create nodes from the results
            nodes = [
                Document(text=doc.page_content, extra_info={"score": doc.metadata.get("score", 1.0)})
                for doc in initial_results
            ]
            
            # Rerank the nodes
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes,
                query_bundle=QueryBundle(question)
            )
            
            # Combine the reranked context
            context = "\n\n".join([node.text for node in reranked_nodes])
            
            # Format the prompt with context and question
            prompt = qa_prompt_tmpl.format(
                context_str=context,
                query_str=question
            )
            
            # Get response from LLM
            response = await self.llm.acomplete(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def close(self):
        """Close connections."""
        if hasattr(self, 'qdrant_client'):
            self.qdrant_client.close()

async def get_response(client, context_str, query_str):
    formatted_prompt = qa_prompt_tmpl.format(
        context_str=context_str,
        query_str=query_str
    )
    
    response = await client.chat(
        model="orca-mini",  # or your chosen model
        messages=[
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]
    )
    
    return response.message.content