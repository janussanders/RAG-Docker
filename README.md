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
