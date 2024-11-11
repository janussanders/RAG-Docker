#!/usr/bin/env python3

import os
from pathlib import Path
from typing import List, Optional
from loguru import logger
from llama_index.core import (
    SimpleDirectoryReader, 
    Document, 
    VectorStoreIndex, 
    StorageContext
)

import pypdf
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
from qdrant_client import QdrantClient

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = '\n'.join(
                page.extract_text() 
                for page in pdf_reader.pages
            )
        logger.info(f"Successfully extracted text from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def process_documents(docs_dir: str = "./docs") -> List[Document]:
    """Process all documents in the specified directory."""
    try:
        reader = SimpleDirectoryReader(
            input_dir=docs_dir,
            filename_as_id=True,
            recursive=True,
            exclude_hidden=True
        )
        documents = reader.load_data()
        logger.info(f"Processed {len(documents)} documents from {docs_dir}")
        return documents
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

def create_chunks(documents: List[Document], chunk_size: int = 1024, overlap: int = 200) -> List[Document]:
    """Split documents into chunks."""
    parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        paragraph_separator="\n\n"
    )
    nodes = parser.get_nodes_from_documents(documents)
    return nodes

def setup_vector_store() -> QdrantVectorStore:
    """Set up connection to Qdrant."""
    client = QdrantClient("localhost", port=6333)
    
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="documents",
        embedding_function=embeddings
    )
    
    return vector_store

def main():
    print("\nDSPy Document Processing System")
    print("=" * 30)
    print("\nThis will process all documents in the ./docs directory and index them in Qdrant.")
    print("Current documents found:")
    
    # List documents that will be processed
    docs_dir = "./docs"
    docs = list(Path(docs_dir).glob("**/*.*"))
    for doc in docs:
        print(f"- {doc.name}")
    
    # Prompt for confirmation
    response = input("\nProceed with processing? (y/n): ").strip().lower()
    if response != 'y':
        print("Operation cancelled.")
        return
    
    print("\nStarting document processing...")
    
    # Process documents
    print("Loading documents...")
    documents = process_documents()
    print(f"Loaded {len(documents)} documents")
    
    # Create chunks
    print("Chunking documents...")
    nodes = create_chunks(documents)
    print(f"Created {len(nodes)} chunks")
    
    # Setup vector store
    print("Setting up vector store...")
    vector_store = setup_vector_store()
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index
    index = VectorStoreIndex.from_documents(nodes, storage_context=storage_context)
    
    print("Document processing completed successfully!")

if __name__ == "__main__":
    main()