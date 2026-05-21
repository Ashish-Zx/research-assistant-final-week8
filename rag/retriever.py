# rag/retriever.py
import re
import json

import numpy as np
from loguru import logger

from rag.embedder import get_embedding
from llm_client import client, get_groq_model

# BM25 state – populated by rag.database.build_bm25_index()
bm25_index = None
chunk_ids: list[str] = []
chunk_docs: list[str] = []
chunk_metadatas: list[dict] = []
tokenized_corpus: list[list[str]] = []


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ----- Self-query filter extraction -----

def extract_query_and_filter(user_query: str) -> dict:
    """Use the LLM to extract a semantic search query and optional metadata filter."""
    prompt = f"""You are a helpful assistant. Given a user question, extract two things:
1. A "query" string optimized for semantic search.
2. A "filter" dictionary for metadata. Only include key: "year" if a specific year is mentioned.

Return only a JSON object. Example:
Question: "What was Nepal's GDP in 2022?"
Output: {{"query": "Nepal GDP 2022", "filter": {{"year": "2022"}}}}

User question: {user_query}
JSON:"""
    resp = client.chat.completions.create(
        model=get_groq_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        logger.warning(f"Self-query parsing failed, using raw query: {user_query}")
        return {"query": user_query, "filter": {}}


# ----- Hybrid search -----

def hybrid_search(
    query: str,
    top_k: int = 10,
    doc_id: str = None,
    where_filter: dict = None,
) -> list[str]:
    """Combine vector search and BM25 via Reciprocal Rank Fusion."""
    from rag.database import collection  # late import to avoid circular deps

    conditions = []
    if doc_id:
        conditions.append({"doc_id": doc_id})
    if where_filter:
        for key, value in where_filter.items():
            conditions.append({key: value})
    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    if bm25_index is None:
        query_emb = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_emb], n_results=top_k, where=where
        )
        return results["documents"][0] if results["documents"] else []

    # Vector search
    query_emb = get_embedding(query)
    vector_results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k * 5,
        where=where,
        include=["documents", "metadatas"],
    )
    vector_ids = vector_results["ids"][0]
    vector_docs = vector_results["documents"][0]
    vect_rank = {
        vid: (rank, doc)
        for rank, (vid, doc) in enumerate(zip(vector_ids, vector_docs), start=1)
    }

    # BM25 search
    tokenized_query = tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    if where_filter:
        for idx, meta in enumerate(chunk_metadatas):
            if not all(meta.get(k) == v for k, v in where_filter.items()):
                bm25_scores[idx] = -1e9
    top_bm25_idx = np.argsort(bm25_scores)[::-1][: top_k * 2]

    # RRF fusion
    k = 60
    fused_scores: dict[str, float] = {}
    for cid, (rank, _) in vect_rank.items():
        fused_scores[cid] = 1.0 / (k + rank)
    for rank, idx in enumerate(top_bm25_idx, start=1):
        cid = chunk_ids[idx]
        if bm25_scores[idx] <= -1:
            continue
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank)

    sorted_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    id_to_doc = dict(zip(vector_ids, vector_docs))
    final_chunks = []
    for cid in sorted_ids:
        if cid in id_to_doc:
            final_chunks.append(id_to_doc[cid])
        else:
            res = collection.get(ids=[cid])
            if res["documents"]:
                final_chunks.append(res["documents"][0])
    return final_chunks


# ----- Re-ranking -----

def rerank_chunks(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """Use the LLM to rerank *chunks* by relevance to *query*."""
    if len(chunks) <= top_k:
        return chunks

    preview_len = 1000
    indexed = "\n".join(
        [f"[{i}] {chunk[:preview_len]}" for i, chunk in enumerate(chunks)]
    )
    prompt = f"""You are a helpful assistant.
Given the question: "{query}"
And the following chunks (each with an index):

{indexed}

Return ONLY a valid JSON object with a key "indices" containing a list of the {top_k} most relevant chunk indices.
Example: {{"indices": [2,5,7]}}
Do not include any other text, code fences, or explanation."""

    resp = client.chat.completions.create(
        model=get_groq_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    raw = resp.choices[0].message.content.strip()
    logger.debug(f"Re-rank raw response: {raw}")

    def extract_indices(text: str) -> list[int]:
        try:
            return json.loads(text)["indices"]
        except Exception:
            pass
        clean = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(clean)["indices"]
        except Exception:
            pass
        match = re.search(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]", clean)
        if match:
            return [int(n) for n in re.findall(r"\d+", match.group(0))]
        nums = [int(n) for n in re.findall(r"\d+", text) if int(n) < len(chunks)]
        return nums[:top_k]

    indices = extract_indices(raw)
    if not indices:
        logger.warning("Re-ranking produced no valid indices, using first chunks")
        return chunks[:top_k]

    valid: list[int] = []
    seen: set[int] = set()
    for i in indices:
        if 0 <= i < len(chunks) and i not in seen:
            valid.append(i)
            seen.add(i)
    if not valid:
        return chunks[:top_k]

    logger.info(f"Re-ranking selected indices: {valid}")
    return [chunks[i] for i in valid]


# ----- Answer generation -----

def generate_answer(query: str, chunks: list[str]) -> str:
    """Generate a grounded answer from retrieved chunks."""
    if not chunks:
        return "No relevant information found."
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful research assistant. Use the provided context to answer the question.
The context may contain table rows, lists, or unstructured text. Extract the relevant numbers or facts directly.

Context:
{context}

Question: {query}
Answer:"""
    resp = client.chat.completions.create(
        model=get_groq_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ----- Document summarisation -----

def summarize_document(top_k: str = "10") -> str:
    """Return a short summary of the most recently indexed document."""
    from rag.database import collection  # late import to avoid circular deps

    dummy_emb = get_embedding("summary overview")
    results = collection.query(query_embeddings=[dummy_emb], n_results=int(top_k))
    chunks = results["documents"][0] if results["documents"] else []
    if not chunks:
        return "No document uploaded yet."
    prompt = f"""You are given excerpts from a document. Write a 2-3 sentence summary describing what the document is about. Focus on the main topic, key entities, and purpose.

Excerpts:
{chr(10).join(['- ' + c[:500] for c in chunks])}

Summary:"""
    resp = client.chat.completions.create(
        model=get_groq_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content
