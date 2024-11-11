import pytest
from pathlib import Path
from src.document_processor import DocumentProcessor
from src.process_docs import process_documents, create_chunks

def test_document_loading():
    processor = DocumentProcessor()
    documents = process_documents()
    assert len(documents) > 0
