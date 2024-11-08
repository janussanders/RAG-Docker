import pytest
import asyncio
import os
from pathlib import Path
from src.document_processor import DocumentProcessor
from llama_index import VectorStoreIndex

@pytest.fixture
async def test_docs_dir(tmp_path):
    """Create a temporary directory with test PDF files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Create a test PDF file
    test_pdf = docs_dir / "test.pdf"
    test_pdf.write_bytes(b"%PDF-1.4\n%Test PDF content")
    
    return str(docs_dir)

@pytest.fixture
async def processor(test_docs_dir):
    """Create and initialize a DocumentProcessor instance."""
    processor = DocumentProcessor(docs_dir=test_docs_dir)
    await processor.initialize()
    yield processor
    await processor.close()  # Cleanup after tests

@pytest.mark.asyncio
async def test_init_collection(processor):
    """Test that collection is initialized properly."""
    collection_info = await processor.get_collection_info()
    assert collection_info.name == 'info_from_docs'
    assert collection_info.vectors_config.size == 384

@pytest.mark.asyncio
async def test_load_documents(processor, test_docs_dir):
    """Test document loading functionality."""
    documents = await processor.load_documents()
    assert len(documents) > 0
    assert all(isinstance(doc, str) for doc in documents)

@pytest.mark.asyncio
async def test_nonexistent_directory():
    """Test handling of nonexistent directory."""
    processor = DocumentProcessor(docs_dir='./nonexistent')
    await processor.initialize()
    with pytest.raises(FileNotFoundError):
        await processor.load_documents()
    await processor.close()

@pytest.mark.asyncio
async def test_empty_directory(tmp_path):
    """Test handling of empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    processor = DocumentProcessor(docs_dir=str(empty_dir))
    await processor.initialize()
    documents = await processor.load_documents()
    assert len(documents) == 0
    await processor.close()

@pytest.mark.asyncio
async def test_create_index(processor, test_docs_dir):
    """Test index creation functionality."""
    # Load documents
    documents = await processor.load_documents()
    
    # Create index
    index = await processor.create_index(documents)
    
    # Verify index was created
    assert isinstance(index, VectorStoreIndex)
    assert processor.index is not None
    
    # Verify documents were embedded
    collection_info = await processor.get_collection_info()
    assert collection_info.points_count > 0

@pytest.mark.asyncio
async def test_create_index_no_documents(processor):
    """Test index creation with no documents."""
    with pytest.raises(FileNotFoundError):
        await processor.create_index()

@pytest.mark.asyncio
async def test_create_index_empty_directory(tmp_path):
    """Test index creation with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    processor = DocumentProcessor(docs_dir=str(empty_dir))
    await processor.initialize()
    
    # Should create empty index without error
    index = await processor.create_index()
    assert isinstance(index, VectorStoreIndex)
    
    await processor.close()
