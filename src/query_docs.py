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
from rich.console import Console
from rich.status import Status
import aioconsole
from asyncio import Queue

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
        self.documents = None
        
        # Initialize clients
        self.qdrant_client = QdrantClient(url="qdrant", port=6333)
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model
        )
        
        # Initialize LLM
        logger.info("Initializing Ollama with orca-mini model...")
        self.llm = Ollama(
            model="orca-mini",
            base_url="http://host.docker.internal:11434",
            request_timeout=180.0
        )
        
        # Initialize reranker
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=3
        )
        
        self.vector_store = None
        self.index = None
        self.query_engine = None
        
        # Store the prompt template as instance variable
        self.qa_prompt = qa_prompt_tmpl
        
    def _initialize_model(self):
        # Initialize your model here
        pass
        
    async def process_documents(self) -> List[Document]:
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
            
            logger.info(f"Starting setup with {len(self.documents)} documents")
            
            # Configure settings
            logger.info("Configuring settings...")
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info("Settings configured successfully")
            
            # Generate UUIDs for documents
            logger.info("Generating UUIDs for documents...")
            for doc in self.documents:
                doc.doc_id = str(uuid.uuid4())
            logger.info("UUIDs generated successfully")
            
            # Verify Qdrant client
            logger.info("Verifying Qdrant client connection...")
            if not self.qdrant_client:
                raise ValueError("Qdrant client not initialized")
            try:
                # Test the connection
                self.qdrant_client.get_collections()
                logger.info("Qdrant client connection verified")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Qdrant: {str(e)}")
            
            # Initialize vector store
            logger.info("Initializing vector store...")
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                metadata_payload_key="metadata"
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create index from documents
            logger.info("Creating vector index from documents...")
            self.index = VectorStoreIndex.from_documents(
                documents=self.documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            # Create query engine with the instance prompt template
            logger.info("Creating query engine...")
            self.query_engine = self.index.as_query_engine(
                text_qa_template=self.qa_prompt
            )
            
            if not self.query_engine:
                raise ValueError("Failed to create query engine")
            logger.info("Query engine created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}")
            raise

    async def query(self, query_text: str) -> Optional[str]:
        """Process a query and return the response."""
        console = Console()
        
        if self.query_engine is None:
            logger.error("Query engine not initialized")
            return "Error: Query engine not initialized"
            
        try:
            logger.info(f"Processing query: {query_text}")
            
            # Try aquery first
            try:
                logger.info("Attempting async query...")
                response = await self.query_engine.aquery(query_text)
                logger.info(f"Async query successful: {str(response)[:100]}...")
                
                # If this was the test query, start interactive mode
                if query_text == "What is DSPy?":
                    await console.aprint("""
=== DSPy Documentation Query System ===
Type 'exit' or 'quit' to end the session
Type 'help' for instructions
======================================
""")
                    
                    # Create message queue for async communication
                    message_queue = Queue()
                    
                    async def input_handler():
                        while True:
                            try:
                                # Use aioconsole for async input
                                user_input = await aioconsole.ainput("> ")
                                await message_queue.put(user_input.strip())
                            except asyncio.CancelledError:
                                break
                    
                    async def query_handler():
                        while True:
                            try:
                                user_query = await message_queue.get()
                                
                                if not user_query:
                                    continue
                                    
                                if user_query.lower() in ['exit', 'quit']:
                                    await console.aprint("\nGoodbye!")
                                    return
                                    
                                if user_query.lower() == 'help':
                                    await console.aprint("""
Instructions:
- Enter your question about DSPy
- Type 'exit' or 'quit' to end the session
- Type 'help' to see these instructions again
""")
                                    continue
                                
                                async with Status("Processing query...", console=console):
                                    query_response = await self.query_engine.aquery(user_query)
                                    if query_response:
                                        await console.aprint(f"\nAnswer: {query_response}\n")
                                    else:
                                        await console.aprint("\nNo response received. Please try again.\n")
                                        
                            except asyncio.CancelledError:
                                break
                            except Exception as e:
                                logger.error(f"Error processing query: {str(e)}")
                                await console.aprint(f"Error: {str(e)}\n")
                return str(response)
                
            except AttributeError:
                # If aquery fails, try regular query
                logger.info("Async query not available, falling back to sync query...")
                response = self.query_engine.query(query_text)
                logger.info(f"Sync query successful: {str(response)[:100]}...")
                return str(response)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error: {str(e)}"

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

    async def test_query_engine(self):
        """Test if the query engine is working."""
        try:
            test_query = "test"
            logger.info("Testing query engine...")
            if not self.query_engine:
                raise ValueError("Query engine not initialized")
            response = await self.query_engine.aquery(test_query)
            logger.info(f"Test query successful: {str(response)[:100]}")
            return True
        except Exception as e:
            logger.error(f"Query engine test failed: {e}")
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

async def get_user_input():
    """Async wrapper for input function"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, "> ")

async def run_interactive_session(querier: DocumentQuerier):
    """Run the interactive query session"""
    try:
        # Test the system
        print("⠸ Testing system...", end='\r')
        test_response = await querier.query("What is DSPy?")
        if test_response and "error" not in test_response.lower():
            print("✓ System is ready")
            logger.success("RAG system initialized successfully!")
            
            # Print welcome message
            print("\n=== DSPy Documentation Query System ===")
            print("Type 'exit' or 'quit' to end the session")
            print("Type 'help' for instructions")
            print("======================================\n")
            
            while True:
                try:
                    # Get user input asynchronously
                    user_query = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
                    
                    if not user_query:
                        continue
                        
                    if user_query.lower() in ['exit', 'quit']:
                        print("\nGoodbye!")
                        break
                        
                    if user_query.lower() == 'help':
                        print("\nInstructions:")
                        print("- Enter your question about DSPy")
                        print("- Type 'exit' or 'quit' to end the session")
                        print("- Type 'help' to see these instructions again\n")
                        continue
                    
                    print("Processing query...", end='\r')
                    response = await querier.query(user_query)
                    print(" " * 50, end='\r')  # Clear processing message
                    
                    if response:
                        print(f"\nAnswer: {response}\n")
                    else:
                        print("\nNo response received. Please try again.\n")
                    
                except asyncio.CancelledError:
                    print("\nOperation cancelled.")
                    break
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in query loop: {str(e)}")
                    print(f"\nError: {str(e)}")
        else:
            logger.error("System test failed!")
            print("✗ System test failed")
            
    except Exception as e:
        logger.exception("Error in interactive session")
        raise
    finally:
        querier.close()