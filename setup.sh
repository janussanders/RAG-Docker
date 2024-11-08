#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print success messages
success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages
error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print warning messages
warn() {
    echo -e "${YELLOW}! $1${NC}"
}

PROJECT_DIR="/Volumes/Algernon/RAG/RAG-Docker"

echo "Starting comprehensive setup..."

# Clean up before setup
echo "Performing pre-setup cleanup..."

# Remove existing environment
rm -rf rag_env
success "Removed old virtual environment"

# Remove symlinks and extended attributes
find "$PROJECT_DIR" -type l -exec rm -f {} \;
success "Removed symlinks"

find "$PROJECT_DIR" -name "._*" -delete
success "Removed extended attribute files"

xattr -cr "$PROJECT_DIR"
success "Removed extended attributes"

# Stop any running containers and clean up Docker
echo "Cleaning Docker environment..."
docker-compose down -v
success "Cleaned Docker environment"

# Deactivate any active virtual environment
deactivate 2>/dev/null || true

# Check for Homebrew Python 3.9
if command -v brew &> /dev/null; then
    if brew list python@3.9 &> /dev/null; then
        PYTHON_PATH="$(brew --prefix python@3.9)/bin/python3.9"
    else
        error "Python 3.9 not found in Homebrew. Please install with: brew install python@3.9"
    fi
else
    error "Homebrew not found. Please install Homebrew first."
fi

echo "Using Python at: $PYTHON_PATH"

# Create new virtual environment
$PYTHON_PATH -m venv rag_env --clear

# Activate the environment
source rag_env/bin/activate

# Create cache directories
mkdir -p "$PROJECT_DIR/cache/transformers"
mkdir -p "$PROJECT_DIR/cache/huggingface"
mkdir -p "$PROJECT_DIR/cache/sentence_transformers"
success "Created cache directories"

# Create and set up Qdrant storage directory
echo "Setting up Qdrant storage..."
mkdir -p "$PROJECT_DIR/qdrant_storage"
chmod 755 "$PROJECT_DIR/qdrant_storage"
success "Created Qdrant storage directory"

# Add to cleanup to ensure proper permissions after cleanup
echo "# Ensure Qdrant storage directory exists with proper permissions" >> rag_env/bin/activate
echo "mkdir -p '$PROJECT_DIR/qdrant_storage'" >> rag_env/bin/activate
echo "chmod 755 '$PROJECT_DIR/qdrant_storage'" >> rag_env/bin/activate

# Set up optimizations for Apple Silicon
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
export CMAKE_ARGS="-DLLAMA_METAL=on"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONIOENCODING=utf8
export TRANSFORMERS_CACHE="$PROJECT_DIR/cache/transformers"
export HF_HOME="$PROJECT_DIR/cache/huggingface"
export SENTENCE_TRANSFORMERS_HOME="$PROJECT_DIR/cache/sentence_transformers"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch first for Apple Silicon
pip install --no-cache-dir torch torchvision torchaudio
success "Installed PyTorch"

# Install transformers and sentence-transformers first
pip install --no-cache-dir transformers
pip install --no-cache-dir sentence-transformers
success "Installed transformer libraries"

# Install core dependencies in order
pip install --no-cache-dir llama-index>=0.11.22
pip install --no-cache-dir llama-index-core>=0.11.22
pip install --no-cache-dir llama-index-llms-ollama>=0.3.6
pip install --no-cache-dir llama-index-embeddings-huggingface>=0.3.1
pip install --no-cache-dir llama-index-vector-stores-qdrant>=0.3.3
success "Installed llama-index packages"

# Install other requirements
pip install --no-cache-dir -r requirements.txt
success "Installed additional requirements"

# Fix permissions
chmod -R 755 rag_env/bin/*

# Add environment variables to activation script
echo "export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1" >> rag_env/bin/activate
echo "export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1" >> rag_env/bin/activate
echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> rag_env/bin/activate
echo "export CMAKE_ARGS='-DLLAMA_METAL=on'" >> rag_env/bin/activate
echo "export PYTHONIOENCODING=utf8" >> rag_env/bin/activate
echo "export TRANSFORMERS_CACHE='$PROJECT_DIR/cache/transformers'" >> rag_env/bin/activate
echo "export HF_HOME='$PROJECT_DIR/cache/huggingface'" >> rag_env/bin/activate
echo "export SENTENCE_TRANSFORMERS_HOME='$PROJECT_DIR/cache/sentence_transformers'" >> rag_env/bin/activate

# Start Qdrant
echo "Setting up Qdrant..."
docker-compose up -d qdrant

# Wait for Qdrant to initialize
echo "Waiting for Qdrant to initialize..."
sleep 10

# Test Qdrant connection
if curl -s -f "http://localhost:6333/collections" > /dev/null; then
    success "Qdrant is running and responding"
else
    error "Qdrant failed to start properly"
    docker logs qdrant
    warn "You may need to run ./start_qdrant.sh manually"
fi

# Create activation script
cat > activate_rag.sh << 'EOL'
#!/bin/bash

# Source the virtual environment
source rag_env/bin/activate

# Check if Qdrant is running
if ! curl -s -f "http://localhost:6333/collections" > /dev/null; then
    echo "Qdrant is not running. Starting Qdrant..."
    docker-compose up -d qdrant
    sleep 5
    if curl -s -f "http://localhost:6333/collections" > /dev/null; then
        echo "✓ Qdrant started successfully"
    else
        echo "! Warning: Qdrant failed to start"
    fi
fi

# Print status
echo "✓ Environment activated with Apple Silicon optimizations"
echo "✓ Metal Performance Shaders (MPS) enabled"
echo "✓ UTF-8 encoding configured"
echo "✓ Cache directories set"
EOL

# Make activation script executable
chmod +x activate_rag.sh

success "Setup complete!"
echo "To activate the environment, use: source ./activate_rag.sh"


