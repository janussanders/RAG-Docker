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

# Add the prompt template as a constant at the top
QA_TEMPLATE = """You are a helpful AI assistant. Use the following context to answer the question. 
If you cannot find the answer in the context, say "I cannot find the answer in the provided context."

Context:
{context_str}

Question: {query_str}

Answer: """

qa_prompt_tmpl = PromptTemplate(template=QA_TEMPLATE)

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
        
        # Verify all components are properly initialized
        self._verify_initialization()

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

    def verify_documents(self):
        """Verify that documents are properly loaded."""
        if not self.documents:
            logger.error("No documents loaded!")
            return False
            
        logger.info(f"Number of documents: {len(self.documents)}")
        for i, doc in enumerate(self.documents):
            logger.info(f"Document {i+1}:")
            logger.info(f"  - ID: {doc.doc_id}")
            logger.info(f"  - Length: {len(doc.text)} characters")
            logger.info(f"  - Preview: {doc.text[:100]}...")
        return True

    async def process_documents(self) -> List[Document]:
        """Load and process documents from the docs directory."""
        try:
            # List all PDF files
            pdf_files = list(self.docs_dir.glob("*.pdf"))
            logger.info(f"Found PDF files: {[f.name for f in pdf_files]}")
            
            if not pdf_files:
                raise FileNotFoundError(f"No PDF files found in {self.docs_dir}")
                
            reader = SimpleDirectoryReader(
                str(self.docs_dir),
                recursive=True,
                filename_as_id=True,
                required_exts=[".pdf"]
            )
            
            self.documents = reader.load_data()
            
            # Verify documents after loading
            if not self.verify_documents():
                raise ValueError("Document verification failed")
                
            return self.documents
            
        except Exception as e:
            logger.exception("Error processing documents")
            raise

    def setup_vector_store(self):
        """Initialize vector store with Qdrant."""
        try:
            if not self.documents:
                raise ValueError("No documents loaded. Call process_documents first.")
                
            logger.info("Setting up vector store...")
            
            # Configure global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            Settings.chunk_size = self.chunk_size
            Settings.chunk_overlap = self.chunk_overlap
            
            # Log collection info
            logger.info(f"Using collection: {self.collection_name}")
            
            # Initialize vector store
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding_function=self.embed_model
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create the base index with logging
            logger.info("Creating vector index from documents...")
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                storage_context=storage_context
            )
            logger.info("Vector index created successfully")
            
            # Configure the query engine with all components
            logger.info("Configuring query engine...")
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=10,  # Get top 10 chunks for reranking
                node_postprocessors=[self.reranker],  # Apply reranking
                text_qa_template=qa_prompt_tmpl,  # Use custom prompt template
                streaming=False  # Disable streaming for simpler response handling
            )
            logger.info("Query engine configured successfully")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

    async def query(self, question: str) -> str:
        """Process a query and return the response."""
        try:
            logger.info(f"Processing query: {question}")
            
            if not self.query_engine:
                raise RuntimeError("Query engine not initialized. Did you call setup_vector_store()?")
                
            if not self.documents:
                raise RuntimeError("No documents loaded. Cannot process query.")
                
            # Log the query attempt
            logger.info("Sending query to engine...")
            
            # The query engine now handles:
            # 1. Vector similarity search (via vector_store)
            # 2. Reranking of results (via reranker)
            # 3. Context assembly and prompt formatting (via text_qa_template)
            # 4. LLM response generation (via llm)
            response = await self.query_engine.aquery(question)
            
            if not response:
                logger.warning("Received empty response from query engine")
                return "I don't know!"
                
            if not response.response:
                logger.warning("Response object has no response text")
                return "I don't know!"
                
            logger.info(f"Query response received: {response.response[:100]}...")
            return response.response
            
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def close(self):
        """Close connections."""
        if hasattr(self, 'qdrant_client'):
            self.qdrant_client.close()

    def _verify_initialization(self):
        """Verify that all required components are initialized."""
        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized")
        if not self.embed_model:
            raise RuntimeError("Embedding model not initialized")
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        if not self.reranker:
            raise RuntimeError("Reranker not initialized")
        logger.info("All components initialized successfully")

    async def test_query_pipeline(self):
        """Test the entire query pipeline."""
        try:
            logger.info("Testing query pipeline...")
            
            # 1. Verify documents
            logger.info("1. Verifying documents...")
            if not self.verify_documents():
                raise RuntimeError("Document verification failed")
                
            # 2. Verify vector store
            logger.info("2. Verifying vector store...")
            if not self.vector_store:
                raise RuntimeError("Vector store not initialized")
                
            # 3. Verify query engine
            logger.info("3. Verifying query engine...")
            if not self.query_engine:
                raise RuntimeError("Query engine not initialized")
                
            # 4. Test simple query
            logger.info("4. Testing simple query...")
            test_query = "What documents are available?"
            response = await self.query(test_query)
            logger.info(f"Test query response: {response}")
            
            logger.info("Query pipeline test completed successfully")
            return True
            
        except Exception as e:
            logger.exception("Query pipeline test failed")
            return False

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