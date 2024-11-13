FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONIOENCODING=utf8
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV CMAKE_ARGS="-DLLAMA_METAL=on"

# Expose both Streamlit and Ollama ports
EXPOSE 8501
EXPOSE 11434

# Create startup script
RUN echo '#!/bin/bash\n\
ollama serve & \
sleep 5 && \
ollama pull orca-mini && \
streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501' > start.sh && \
chmod +x start.sh

# Run the startup script instead of python -m
CMD ["./start.sh"] 