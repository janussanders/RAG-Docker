#!/usr/bin/env python3

import os
import torch
import time
from pathlib import Path
from loguru import logger
from typing import Optional
import logging
from rich.console import Console
from rich.spinner import Spinner
from rich import print as rprint
from fastapi import FastAPI
from transformers import logging as tf_logging
from huggingface_hub import logging as hf_logging

# Enable Metal optimizations for Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("✓ Using Metal Performance Shaders (MPS) for acceleration")
else:
    device = torch.device("cpu")
    logger.info("! Using CPU (MPS not available)")

# Disable progress bars and reduce logging noise
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
tf_logging.set_verbosity_error()
hf_logging.set_verbosity_error()

from .query_docs import DocumentQuerier

# Setup file handler for health checks
health_check_logger = logging.getLogger('health_checks')
health_check_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('logs/health_checks.log')
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
health_check_logger.addHandler(file_handler)

console = Console()

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

async def wait_for_services(timeout: int = 30):
    """Wait for services to be ready"""
    import aiohttp
    from datetime import datetime
    
    services = {
        'Qdrant': 'http://qdrant:6333/',
        'Ollama': 'http://ollama:11434/api/version'
    }
    
    console.print("[yellow]Waiting for services to initialize...[/yellow]")
    
    async with aiohttp.ClientSession() as session:
        for service, url in services.items():
            attempt = 1
            while True:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            console.print(f"[green]✓ {service} is ready[/green]")
                            break
                except aiohttp.ClientError:
                    if attempt >= timeout:
                        error_msg = f"ERROR: {service} is not available after {timeout} seconds"
                        raise RuntimeError(error_msg)
                    attempt += 1
                    await asyncio.sleep(1)
                    if attempt % 5 == 0:  # Only print every 5 attempts
                        console.print(f"[yellow]Still waiting for {service}... (attempt {attempt})[/yellow]")

async def interactive_session(querier):
    """Run an interactive query session in the terminal with rich formatting."""
    # Clear the screen first
    console.clear()
    
    rprint("\n[bold blue]=== DSPy Documentation Query System ===[/bold blue]")
    rprint("[dim]Type 'exit' or 'quit' to end the session")
    rprint("[dim]Type 'help' for instructions[/dim]")
    rprint("[blue]======================================[/blue]\n")

    while True:
        try:
            # Get user input
            query = console.input("\n[bold green]Enter your question:[/bold green] ").strip()
            
            if query.lower() in ['exit', 'quit']:
                rprint("\n[yellow]Goodbye![/yellow]")
                break
            
            if query.lower() == 'help':
                rprint("\n[bold]Instructions:[/bold]")
                rprint("- Ask any question about the DSPy documentation")
                rprint("- Questions can be about concepts, usage, or examples")
                rprint("- Type 'exit' or 'quit' to end the session")
                continue
            
            if not query:
                continue
            
            with console.status("[bold yellow]Searching for answer...[/bold yellow]"):
                response = await querier.query(query)
            
            rprint("\n[bold]Answer:[/bold]")
            rprint(f"{response}\n")
            rprint("[dim]-------------------------------------------[/dim]")

        except KeyboardInterrupt:
            rprint("\n\n[yellow]Session terminated by user. Goodbye![/yellow]")
            break
        except Exception as e:
            rprint(f"\n[red]Error: {str(e)}[/red]")
            rprint("[dim]Please try again or type 'exit' to quit.[/dim]")

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
        
        # Start both the API server and interactive mode if requested
        if args.interactive:
            import uvicorn
            server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="error"))
            await asyncio.gather(
                server.serve(),
                interactive_session(querier)
            )
        
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
