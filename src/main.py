#!/usr/bin/env python3

import os
import torch
import time
from pathlib import Path
from loguru import logger
from typing import Optional

# Enable Metal optimizations for Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("✓ Using Metal Performance Shaders (MPS) for acceleration")
else:
    device = torch.device("cpu")
    logger.info("! Using CPU (MPS not available)")

from .document_processor import DocumentProcessor
from .process_docs import process_documents, create_chunks, setup_vector_store
from .query_docs import DocumentQuerier

async def wait_for_services(timeout: int = 30):
    """Wait for Qdrant and Ollama to be ready"""
    import aiohttp
    import asyncio
    
    services = {
        'Qdrant': 'http://localhost:6333/health',
        'Ollama': 'http://localhost:11434/api/tags'
    }
    
    async with aiohttp.ClientSession() as session:
        for service, url in services.items():
            logger.info(f"Waiting for {service}...")
            for _ in range(timeout):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            logger.success(f"✓ {service} is ready")
                            break
                except aiohttp.ClientError:
                    await asyncio.sleep(1)
            else:
                raise RuntimeError(f"{service} is not available after {timeout} seconds")

async def main():
    try:
        # Wait for services
        await wait_for_services()
        
        # Initialize document processor
        processor = DocumentProcessor()
        await processor.initialize()
        
        # Process documents
        documents = await processor.load_documents()
        
        # Create chunks and index
        chunks = create_chunks(documents)
        vector_store = setup_vector_store()
        
        # Initialize querier
        querier = DocumentQuerier()
        
        # Run test query
        test_query = "What is DSPy and what are its main features?"
        response = querier.query(test_query)
        querier.print_results(response)
        
        logger.success("RAG system initialized and tested successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'processor' in locals():
            await processor.close()
        if 'querier' in locals():
            querier.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
