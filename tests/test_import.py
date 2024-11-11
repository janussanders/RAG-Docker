# Import only what's needed for memory tracking
import psutil
import gc

def print_memory_usage():
    memory = psutil.Process().memory_info().rss / (1024 * 1024)
    available = psutil.virtual_memory().available / (1024 * 1024)
    total = psutil.virtual_memory().total / (1024 * 1024)
    print(f"Memory usage: {memory:.2f} MB")
    print(f"Available system memory: {available:.2f} MB")
    print(f"Total system memory: {total:.2f} MB")

print("\nInitial memory usage:")
print_memory_usage()

# Clear memory before imports
gc.collect()
print("\nMemory cleared")
print_memory_usage()

# Test specific imports instead of importing everything
print("\nTesting specific imports:")
try:
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    print("Success: QdrantVectorStore import")
except ImportError:
    print("Failed: QdrantVectorStore import")

gc.collect()
print("\nFinal memory usage:")
print_memory_usage()

def test_src_imports():
    """Test that all source modules can be imported"""
    from src import document_processor
    from src import process_docs
    from src import query_docs
    from src import main
    from src import api
    
    assert True 