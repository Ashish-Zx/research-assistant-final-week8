# rag/chunker.py
import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(file_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_text_from_txt(file_path: str) -> str:
    """Read plain-text (or markdown) file content."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(
    text: str, chunk_size: int = 200, overlap: int = 50
) -> list[str]:
    """Split *text* into overlapping word-level chunks."""
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks
