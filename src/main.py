#!/usr/bin/env python3

import os
import torch
import time
from pathlib import Path
from loguru import logger
from typing import Optional
import logging

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

# Setup file handler for health checks
health_check_logger = logging.getLogger('health_checks')
health_check_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logs/health_checks.log')
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
health_check_logger.addHandler(file_handler)

async def wait_for_services(timeout: int = 30):
    """Wait for services to be ready"""
    import aiohttp
    from datetime import datetime
    
    log_file = "logs/health_checks.log"
    
    def log_message(msg: str):
        timestamp = datetime.now().isoformat()
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {msg}\n")
    
    services = {
        'Qdrant': 'http://qdrant:6333/',
        'Ollama': 'http://ollama:11434/api/version'
    }
    
    async with aiohttp.ClientSession() as session:
        for service, url in services.items():
            log_message(f"Waiting for {service}...")
            attempt = 1
            while True:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            log_message(f"✓ {service} is ready (attempt {attempt})")
                            break
                except aiohttp.ClientError:
                    if attempt >= timeout:
                        error_msg = f"ERROR: {service} is not available after {timeout} seconds"
                        log_message(error_msg)
                        raise RuntimeError(error_msg)
                    attempt += 1
                    await asyncio.sleep(1)

async def interactive_mode(querier):
    print("\nRAG System Ready! Enter your questions (type 'exit' to quit):")
    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            # Call the query method from the querier instance
            response = await querier.query(question)  # Make sure to await the query
            print(f"\nAnswer: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

async def main():
    try:
        # Wait for services
        await wait_for_services()
        
        # Initialize document querier
        querier = DocumentQuerier()
        
        # Process documents and setup vector store
        documents = await querier.process_documents()
        vector_store = querier.setup_vector_store()
        
        logger.success("RAG system initialized successfully!")
        
        # Start interactive mode if requested
        if args.interactive:
            await interactive_mode(querier)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'querier' in locals():
            querier.close()

def get_prompt_template():
    return """You are an AI assistant helping users understand technical documentation. 
Analyze the following context and question carefully.

CONTEXT:
{context}

QUESTION:
{question}

GUIDELINES:
- Answer based ONLY on the provided context
- If uncertain, express your uncertainty
- If information is missing, say so
- Use specific references from the context
- Keep responses clear and focused

RESPONSE FORMAT:
1. Direct Answer: [Provide concise answer]
2. Supporting Evidence: [Quote relevant context]
3. Additional Context: [If needed, provide clarification]

Answer: """

if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main())
