import pytest
from src.query_docs import DocumentQuerier
from loguru import logger

def test_rag_queries(document_querier):
    """Test specific queries about the DSPy document"""
    test_questions = [
        "What is DSPy?",
        "What are the main components of DSPy?",
        "How does DSPy handle prompting?"
    ]
    
    for question in test_questions:
        response = document_querier.query(question)
        assert response is not None 