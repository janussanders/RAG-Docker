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
from asyncio import Queue
from rich.live import Live
from rich.console import Console
from rich.status import Status
import aioconsole

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

from .query_docs import DocumentQuerier, run_interactive_session

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
    console = Console()
    
    try:
        # Create async status
        async with Status("[yellow]Testing system...", console=console) as status:
            if not querier.query_engine:
                raise RuntimeError("Query engine not initialized")
            if not querier.vector_store:
                raise RuntimeError("Vector store not initialized")
                
            response = await querier.query("What is DSPy?")
            if not response:
                raise RuntimeError("Empty response from query system")
            
            await console.aprint("[green]✓ System is ready[/green]")
            
            # Use rich for better async display
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
                    # Use aioconsole for async input
                    user_query = await aioconsole.ainput("> ")
                    await message_queue.put(user_query.strip())
            
            async def query_handler():
                while True:
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
                    
                    print("Processing query...", end='\r')
                    response = await querier.query(user_query)
                    print(" " * 50, end='\r')  # Clear processing message
                    
                    if response:
                        print(f"\nAnswer: {response}\n")
                    else:
                        print("\nNo response received. Please try again.\n")
                    
                    logger.error(f"Error in query loop: {str(e)}")
                    print(f"\nError: {str(e)}")
            
            # Run the input processing loop
            await input_handler()
            await query_handler()
            return True
            
    except Exception as e:
        console.print(f"[red]✗ System check failed: {str(e)}[/red]")
        logger.error(f"System check failed: {e}")
        return False

async def start_rag_system():
    """Initialize and start the RAG system"""
    try:
        # Initialize the querier and pass control to its main loop
        from src.query_docs import DocumentQuerier, run_interactive_session
        
        querier = DocumentQuerier()
        await querier.process_documents()
        querier.setup_vector_store()
        
        # Hand off control to the interactive session
        await run_interactive_session(querier)
        
    except Exception as e:
        logger.exception("Error starting RAG system")
        raise

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_rag_system())
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        loop.close()
