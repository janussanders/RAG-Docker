moving-readme-file
# RAG-powered DSPy Documentation Assistant

A Docker-based RAG (Retrieval Augmented Generation) system that provides intelligent querying of DSPy documentation using Ollama, Qdrant, and LlamaIndex.

## 🚀 Features

- **RAG Implementation**: Uses LlamaIndex for document processing and retrieval
- **Vector Storage**: Qdrant for efficient vector storage and similarity search
- **Local LLM**: Ollama integration with orca-mini model
- **Optimizations**:
  - Sentence-level chunking with overlap
  - Cross-encoder reranking
  - Custom prompt templates
  - Efficient vector similarity search
  - Asynchronous query processing

## 🛠 Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama Desktop](https://ollama.ai)

## 🔧 System Architecture
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
│ RAG App │ │ Qdrant │ │ Ollama │
│ - LlamaIndex │────▶│ Vector DB │ │ Local LLM │
│ - HF Embeddings│◀────│ │ │
└─────────────────┘ └──────────────┘ └──────────────┘

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Start Ollama Desktop application

3. Pull the required model:
```bash
ollama pull orca-mini
```

4. Build and run with Docker Compose:
```bash
docker compose up --build
```

## 🔍 Optimizations

### Vector Store (Qdrant)
- Persistent storage for document embeddings
- Efficient similarity search
- Scalable vector database

### Document Processing
- Sentence-level chunking (1024 tokens with 200 token overlap)
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Filename-based document tracking

### Query Pipeline
- Cross-encoder reranking (ms-marco-MiniLM-L-2-v2)
- Custom prompt templates for consistent responses
- Top-k retrieval with reranking
- Async query processing

### LLM Integration
- Local inference with orca-mini
- Optimized for Apple Silicon
- Configurable timeout and retry logic

## 🚀 Usage

1. Place PDF documentation in the `docs` directory
2. Start the system:
```bash
docker compose up
```

3. Interact with the CLI interface:
```
=== DSPy Documentation Query System ===
Type 'exit' or 'quit' to end the session
Type 'help' for instructions
======================================

> your question here
```

## 🔧 Configuration

Key configuration parameters in `docker-compose.yml`:
```yaml
environment:
  - CHUNK_SIZE=1024
  - CHUNK_OVERLAP=200
  - COLLECTION_NAME=dspy_docs
  - EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 🏗 Architecture Details

### RAG Application
- Built with Python 3.9
- Uses LlamaIndex for document processing
- Async support for concurrent operations
- Comprehensive logging and error handling

### Vector Store
- Qdrant for vector similarity search
- Persistent storage across sessions
- Efficient index management

### LLM Integration
- Ollama for local inference
- Optimized for Apple Silicon
- Configurable model parameters

## 🐛 Troubleshooting

1. **Ollama Connection Issues**
   - Ensure Ollama Desktop is running
   - Check the model is pulled: `ollama list`
   - Verify host.docker.internal is accessible

2. **Memory Issues**
   - Adjust Docker resource limits
   - Modify chunk size and overlap
   - Consider using a smaller model

3. **Performance Issues**
   - Enable MPS acceleration
   - Adjust batch sizes
   - Monitor resource usage

## 🤝 Special Thanks
Akshay Pachaar, Avi Chawla on their tutorial [https://www.dailydoseofds.com/a-crash-course-on-building-rag-systems-part-1-with-implementations/](A Crash Course on Building RAG Systems – Part 1 (With Implementation))


=======
