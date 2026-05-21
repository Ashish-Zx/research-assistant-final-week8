# rag/database.py
import os
import re
import uuid
import tempfile

import chromadb
from rank_bm25 import BM25Okapi
from loguru import logger

from rag.embedder import get_embedding
from rag.chunker import chunk_text, extract_text_from_pdf, extract_text_from_txt
from config import CHROMA_DB_PATH

# ----- ChromaDB setup -----
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="research_assistant")


def build_bm25_index() -> None:
    """Rebuild the in-memory BM25 index from the current ChromaDB collection."""
    import rag.retriever as retriever

    try:
        results = collection.get(include=["documents", "metadatas"])
        if not results["documents"]:
            retriever.bm25_index = None
            return
        retriever.chunk_ids = results["ids"]
        retriever.chunk_docs = results["documents"]
        retriever.chunk_metadatas = results["metadatas"]
        retriever.tokenized_corpus = [
            retriever.tokenize(doc) for doc in retriever.chunk_docs
        ]
        retriever.bm25_index = BM25Okapi(retriever.tokenized_corpus)
        logger.info(f"BM25 index built with {len(retriever.chunk_docs)} chunks")
    except Exception as e:
        logger.warning(f"Failed to build BM25 index: {e}")
        retriever.bm25_index = None


def process_document(file_content: bytes, filename: str) -> str:
    """Ingest a document: extract text, chunk, embed, and store in ChromaDB."""
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            text = extract_text_from_pdf(tmp_path)
        else:
            text = extract_text_from_txt(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not text.strip():
        raise ValueError("No text found in document")

    chunks = chunk_text(text)
    doc_id = str(uuid.uuid4())
    for i, chunk in enumerate(chunks):
        year_match = re.search(r"\b(19|20)\d{2}\b", chunk)
        year = year_match.group() if year_match else ""
        emb = get_embedding(chunk)
        collection.add(
            ids=[f"{doc_id}_{i}"],
            embeddings=[emb],
            metadatas=[
                {
                    "doc_id": doc_id,
                    "source": filename,
                    "chunk_index": i,
                    "year": year,
                }
            ],
            documents=[chunk],
        )
    build_bm25_index()
    return doc_id
