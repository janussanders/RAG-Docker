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

from .query_docs import DocumentQuerier

async def wait_for_services(timeout: int = 30):
    """Wait for Qdrant (containerized) and Ollama (local on Mac) to be ready"""
    import aiohttp
    import asyncio
    
    services = {
        'Qdrant': 'http://qdrant:6333/',           # Just check root endpoint
        'Ollama': 'http://host.docker.internal:11434/api/tags'  # Mac host machine
    }
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        for service, url in services.items():
            logger.info(f"Waiting for {service}...")
            for attempt in range(timeout):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            logger.success(f"✓ {service} is ready (attempt {attempt + 1})")
                            break
                except aiohttp.ClientError as e:
                    logger.debug(f"Attempt {attempt + 1} failed for {service}: {str(e)}")
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"Unexpected error connecting to {service}: {str(e)}")
                    await asyncio.sleep(1)
            else:
                raise RuntimeError(f"{service} is not available after {timeout} seconds")

async def main():
    try:
        # Wait for services
        await wait_for_services()
        
        # Initialize document querier
        querier = DocumentQuerier()
        
        # Process documents and setup vector store
        documents = await querier.process_documents()
        vector_store = querier.setup_vector_store()
        
        # Run test query
        test_query = "What is DSPy and what are its main features?"
        response = querier.query(test_query)
        logger.info(f"Query response: {response}")
        
        logger.success("RAG system initialized and tested successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'querier' in locals():
            querier.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
