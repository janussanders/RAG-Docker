FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONIOENCODING=utf8
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV CMAKE_ARGS="-DLLAMA_METAL=on"
EXPOSE 8000
CMD ["python", "-m", "src.main"]
