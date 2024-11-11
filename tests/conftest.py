import pytest
import asyncio
import os
from pathlib import Path
from typing import Generator, AsyncGenerator
import tempfile
import shutil

from src.document_processor import DocumentProcessor

# Environment configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "ollama")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

# Load test documents from files
TEST_DOCS_DIR = Path(__file__).parent / "test_documents"
TEST_DOCUMENTS = {}

for file_path in TEST_DOCS_DIR.glob("*"):
    if file_path.suffix == '.pdf':
        TEST_DOCUMENTS[file_path.name] = file_path.read_bytes()
    else:
        TEST_DOCUMENTS[file_path.name] = file_path.read_text()

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def test_docs_dir() -> AsyncGenerator[str, None]:
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "test_docs"
        docs_dir.mkdir(parents=True)
        
        # Create test documents
        for filename, content in TEST_DOCUMENTS.items():
            file_path = docs_dir / filename
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content.strip())
        
        yield str(docs_dir)

@pytest.fixture(scope="function")
async def processor(test_docs_dir: str) -> AsyncGenerator[DocumentProcessor, None]:
    """Create and initialize a DocumentProcessor instance."""
    processor = DocumentProcessor(
        docs_dir=test_docs_dir,
        collection_name="test_collection",
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        ollama_host=OLLAMA_HOST,
        ollama_port=OLLAMA_PORT
    )
    
    try:
        await processor.initialize()
        yield processor
    finally:
        await processor.cleanup() 