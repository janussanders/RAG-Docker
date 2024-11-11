import sys
from pathlib import Path
import pytest

# Add src to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

@pytest.fixture
def document_querier():
    from src.query_docs import DocumentQuerier
    querier = DocumentQuerier()
    yield querier
    querier.close()

@pytest.fixture
def vector_store():
    from src.process_docs import setup_vector_store
    store = setup_vector_store()
    return store