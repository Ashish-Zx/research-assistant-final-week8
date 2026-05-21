# agent/memory.py
import uuid
import chromadb
from datetime import datetime

from rag.embedder import get_embedding
from llm_client import client, get_groq_model
from config import CHROMA_DB_PATH

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
memory_collection = chroma_client.get_or_create_collection(name="conversation_memory")


def store_memory(user_id: str, query: str, answer: str) -> str:
    """Summarise and persist an interaction to the memory collection."""
    summary_prompt = (
        f"Summarize this interaction in one sentence: "
        f"User: {query} Assistant: {answer}"
    )
    resp = client.chat.completions.create(
        model=get_groq_model(),
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=100,
    )
    summary = resp.choices[0].message.content.strip()

    emb = get_embedding(summary)
    memory_collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[emb],
        metadatas=[
            {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "type": "summary",
            }
        ],
        documents=[summary],
    )
    return summary


def retrieve_memories(user_id: str, query: str, top_k: int = 3) -> list[str]:
    """Return the most relevant past interaction summaries for a user."""
    q_emb = get_embedding(query)
    results = memory_collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"user_id": user_id},
    )
    return results["documents"][0] if results["documents"] else []
