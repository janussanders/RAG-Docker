#!/usr/bin/env python3

import logging
from pathlib import Path
import torch
import streamlit as st
from loguru import logger
import sys
import os

# Configure logging
logger.remove()
logger.add(
    "app.log",
    rotation="500 MB",
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    enqueue=True
)
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    enqueue=True
)

# Check for MPS (Metal Performance Shaders) availability
if torch.backends.mps.is_available():
    logger.info("Using MPS (Metal Performance Shaders)")
    device = torch.device("mps")
else:
    logger.info("Using CPU (MPS not available)")
    device = torch.device("cpu")

# Import our Streamlit app
from src.app import StreamlitApp
from src.query_docs import DocumentQuerier

# Environment setup (optional but recommended)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def main():
    """Main entry point for the application"""
    try:
        docs_dir = Path("./docs")
        if not docs_dir.exists():
            logger.error(f"Documents directory not found: {docs_dir}")
            st.error("Documents directory not found. Please ensure the docs folder exists.")
            return

        app = StreamlitApp()
        app.render()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
