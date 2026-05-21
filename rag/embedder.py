# rag/embedder.py
import os

# Keep the model inside the project folder so it survives container restarts
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

USE_OLLAMA = os.getenv("EMBEDDING_PROVIDER", "transformers").lower() == "ollama"

if USE_OLLAMA:
    from openai import OpenAI

    local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def get_embedding(text: str) -> list[float]:
        """Return a 768-dimensional embedding vector via Ollama/nomic-embed-text."""
        resp = local_client.embeddings.create(model="nomic-embed-text", input=text)
        return resp.data[0].embedding

else:
    from sentence_transformers import SentenceTransformer

    # Load the model once at module level (first call downloads ~80 MB)
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(text: str) -> list[float]:
        """Return a 384-dimensional embedding vector via all-MiniLM-L6-v2."""
        return _model.encode(text).tolist()
