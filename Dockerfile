FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the entire project
COPY . .

# Create directories for data and model cache
RUN mkdir -p /app/data /app/models

# Expose the port Render expects (default 8000)
EXPOSE 8000

# Set environment defaults
ENV CHROMA_DB_PATH=/app/data/chroma_db
ENV TRACE_DB_PATH=/app/data/traces.db
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]