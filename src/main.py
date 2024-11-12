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
    """Wait for Qdrant (containerized) to be ready"""
    import aiohttp
    import asyncio
    from datetime import datetime
    
    # Setup health check logging
    log_file = Path("logs/health_checks.log")
    log_file.parent.mkdir(exist_ok=True)
    
    async def log_health_check(message: str):
        timestamp = datetime.now().isoformat()
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    services = {
        'Qdrant': 'http://qdrant:6333/',           
        'Ollama': 'http://ollama:11434/api/version'  # Change to correct health check endpoint
    }
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        for service, url in services.items():
            await log_health_check(f"Waiting for {service}...")
            for attempt in range(timeout):
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            await log_health_check(f"✓ {service} is ready (attempt {attempt + 1})")
                            break
                except aiohttp.ClientError as e:
                    await log_health_check(f"Attempt {attempt + 1} failed for {service}: {str(e)}")
                    await asyncio.sleep(1)
                except Exception as e:
                    await log_health_check(f"Unexpected error connecting to {service}: {str(e)}")
                    await asyncio.sleep(1)
            else:
                error_msg = f"{service} is not available after {timeout} seconds"
                await log_health_check(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)

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
