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
from logging.handlers import RotatingFileHandler
import sys
from contextlib import contextmanager
from io import StringIO
import asyncio

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

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Setup different loggers
def setup_loggers():
    # Main application logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    app_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    app_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s')
    )
    app_logger.addHandler(app_handler)

    # Model download logger
    model_logger = logging.getLogger('model_downloads')
    model_logger.setLevel(logging.INFO)
    model_handler = RotatingFileHandler(
        'logs/model_downloads.log',
        maxBytes=1024*1024,
        backupCount=3
    )
    model_handler.setFormatter(
        logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s')
    )
    model_logger.addHandler(model_handler)

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Initialize console
console = Console()

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

def print_help():
    """Print help information for interactive mode."""
    help_text = """
Available Commands:
------------------
help        Show this help message
exit/quit   Exit the program

Query Examples:
--------------
> What is DSPy?
> How does DSPy handle prompts?
> What are DSPy's main features?

Tips:
-----
- Questions should be clear and specific
- Wait for the response before typing next query
- First query might take longer (model loading)
"""
    print(help_text)

async def interactive_session(querier: DocumentQuerier):
    """Run an interactive query session."""
    print("\n=== DSPy Documentation Query System ===")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'help' for instructions")
    print("======================================\n")

    while True:
        try:
            # Get user input with a prompt
            question = input("> ").strip()
            
            # Handle exit commands
            if question.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if question.lower() == 'help':
                print_help()
                continue
                
            if not question:
                continue

            # Show a spinner while processing
            with console.status("[bold yellow]Processing query...") as status:
                try:
                    # Process the query asynchronously with a timeout
                    response = await asyncio.wait_for(
                        querier.query(question),
                        timeout=30.0  # 30 second timeout
                    )
                    print(f"\n[bold green]Answer:[/bold green] {response}\n")
                except asyncio.TimeoutError:
                    print("[bold red]Error:[/bold red] Query timed out. Please try again.")
                except Exception as e:
                    print(f"[bold red]Error:[/bold red] {str(e)}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive session: {e}")
            print(f"[bold red]Error:[/bold red] {str(e)}")

@contextmanager
def suppress_stdout():
    """Context manager to temporarily suppress stdout"""
    stdout = sys.stdout
    stderr = sys.stderr
    # Redirect stdout/stderr to StringIO
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield
    finally:
        # Restore original stdout/stderr
        sys.stdout = stdout
        sys.stderr = stderr

async def check_system_ready(querier: DocumentQuerier) -> bool:
    """Check if the RAG system is ready to handle queries."""
    try:
        with console.status("[yellow]Testing system..."):
            # Verify query engine exists
            if not querier.query_engine:
                raise RuntimeError("Query engine not initialized")
                
            # Verify vector store exists
            if not querier.vector_store:
                raise RuntimeError("Vector store not initialized")
                
            # Try a simple test query
            response = await querier.query("What is DSPy?")
            if not response:
                raise RuntimeError("Empty response from query system")
                
            console.print("[green]✓ System is ready[/green]")
            return True
    except Exception as e:
        console.print(f"[red]✗ System check failed: {str(e)}[/red]")
        logger.error(f"System check failed: {e}")
        return False

async def main():
    try:
        # Wait for services
        await wait_for_services()
        
        # Initialize document querier
        querier = None
        with suppress_stdout():
            querier = DocumentQuerier()
            
            # Process documents first
            logger.info("Processing documents...")
            documents = await querier.process_documents()
            if not documents:
                raise RuntimeError("No documents processed")
                
            # Setup vector store
            logger.info("Setting up vector store...")
            vector_store = querier.setup_vector_store()
            if not vector_store:
                raise RuntimeError("Vector store setup failed")
        
        # Check if system is ready
        if not await check_system_ready(querier):
            logger.error("System failed readiness check")
            return
            
        logger.success("RAG system initialized successfully!")
        
        # Start interactive session
        if args.interactive:
            await interactive_session(querier)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        # Cleanup
        if querier:
            querier.close()

def get_prompt_template():
    """Get the prompt template for the RAG system."""
    return """You are a helpful AI assistant explaining the DSPy framework and its documentation.
Answer questions based ONLY on the provided context. Be concise and specific.

Context:
{context}

Question:
{question}

Instructions:
1. Use only information from the context
2. If unsure, acknowledge uncertainty
3. If context doesn't contain relevant info, say so
4. Keep responses clear and focused
5. Use specific examples from context when available

Answer: """

if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main())
