# api/main.py
from datetime import date

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from agent.loop import run_agent
from agent.memory import retrieve_memories
from agent.tools.builtins import calculator, get_weather
from agent.tools.events import add_event, list_events
from agent.tools.files import list_files, read_file, write_file
from agent.tools.registry import Tool, ToolRegistry
from agent.tools.currency import convert_currency
from agent.tools.news import get_news
from agent.tools.web import web_search
from agent.tools.wikipedia import search_wikipedia
from api.models import AgentRequest, RatingRequest
from config import LOG_LEVEL
from eval.tracer import add_eval_columns, init_db
from rag.database import build_bm25_index, process_document
from rag.retriever import (
    extract_query_and_filter,
    generate_answer,
    hybrid_search,
    rerank_chunks,
    summarize_document,
)

app = FastAPI(title="Research Assistant")
logger.add("logs/app.log", rotation="1 day", level=LOG_LEVEL)

document_uploaded: bool = False
latest_doc_id: str | None = None

# ----- Tool registry -----
tool_registry = ToolRegistry()


def get_today() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return date.today().isoformat()


def _create_knowledge_base_tool():
    """Factory that closes over the module-level latest_doc_id."""

    def search_knowledge_base(query: str) -> str:
        qf = extract_query_and_filter(query)
        search_text = qf["query"]
        where_filter = qf.get("filter") or {}
        logger.info(f"Self-query filter for '{query}': {where_filter}")

        doc_id = latest_doc_id
        chunks = hybrid_search(
            search_text, where_filter=where_filter, doc_id=doc_id, top_k=20
        )
        if not chunks and where_filter:
            logger.info("No chunks with filter, retrying without filter")
            chunks = hybrid_search(
                search_text, where_filter=None, doc_id=doc_id, top_k=20
            )
        if not chunks:
            return "No relevant information found in the uploaded documents."
        chunks = rerank_chunks(search_text, chunks, top_k=5)
        logger.info(f"Retrieved chunks: {chunks}")
        return generate_answer(search_text, chunks)

    return search_knowledge_base


kb_tool = _create_knowledge_base_tool()

for name, description, func in [
    ("calculator", "Evaluate a mathematical expression.", calculator),
    ("get_weather", "Get current weather for a city.", get_weather),
    ("web_search", "Search the web for up-to-date information.", web_search),
    ("search_knowledge_base", "Search uploaded document for an answer.", kb_tool),
    (
        "summarize_document",
        "Summarize the document that has already been uploaded to the knowledge base. "
        "Call this tool directly when the user asks to summarize, overview, or describe the document — "
        "no extra input is needed.",
        summarize_document,
    ),
    ("read_file", "Read a file from the workspace.", read_file),
    ("write_file", "Write content to a file in the workspace.", write_file),
    ("list_files", "List all files in the workspace.", list_files),
    ("add_event", "Add an event to the local calendar. Date format: YYYY-MM-DD.", add_event),
    ("list_events", "List events, optionally for a specific date (YYYY-MM-DD).", list_events),
    ("get_today", "Get today's date.", get_today),
    ("search_wikipedia", "Search Wikipedia for a topic summary.", search_wikipedia),
    (
        "convert_currency",
        "Convert an amount from one currency to another. Parameters: amount, from_currency (3-letter code), to_currency (3-letter code).",
        convert_currency,
    ),
    ("get_news", "Get recent news headlines on a topic.", get_news),
]:
    tool_registry.register(Tool(name, description, func))


def get_tools_list() -> list[dict]:
    return tool_registry.get_tools_list_for_api()


def get_tools_map() -> dict:
    return {tool.name: tool.func for tool in tool_registry.get_all()}


# ----- Lifecycle -----

@app.on_event("startup")
async def startup_event():
    global document_uploaded, latest_doc_id
    init_db()
    add_eval_columns()
    build_bm25_index()
    document_uploaded = False
    latest_doc_id = None


# ----- Endpoints -----

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global document_uploaded, latest_doc_id
    if file.filename is None:
        raise HTTPException(400, "No file name")
    content = await file.read()
    try:
        doc_id = process_document(content, file.filename)
        document_uploaded = True
        latest_doc_id = doc_id
        return {"doc_id": doc_id, "message": f"Document '{file.filename}' indexed"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Processing failed: {e}")


@app.post("/ask")
async def ask_rag(req: AgentRequest):
    qf = extract_query_and_filter(req.query)
    search_text = qf["query"]
    where_filter = qf.get("filter") or {}
    candidates = hybrid_search(
        search_text, top_k=20, where_filter=where_filter, doc_id=latest_doc_id
    )
    top_chunks = rerank_chunks(search_text, candidates, top_k=5)
    answer = generate_answer(search_text, top_chunks)
    return {"answer": answer}


@app.post("/agent")
async def agent_endpoint(req: AgentRequest, user_id: str = "default_user"):
    memories = retrieve_memories(user_id, req.query)
    memory_context = "\n".join(memories[:3]) if memories else ""

    # Tell the agent a document is available so it doesn't refuse to summarize
    doc_context = (
        "A document has already been uploaded to the knowledge base. "
        "You can call summarize_document or search_knowledge_base directly without asking the user to upload anything.\n\n"
        if document_uploaded
        else ""
    )

    augmented_query = (
        f"{doc_context}Previous relevant memories:\n{memory_context}\n\nCurrent question: {req.query}"
        if memory_context
        else f"{doc_context}{req.query}"
    )
    return StreamingResponse(
        run_agent(augmented_query, get_tools_list(), get_tools_map(), user_id=user_id),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/documents")
async def list_documents():
    from rag.database import collection
    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return []
    # Extract unique source names
    sources = list(set(m["source"] for m in results["metadatas"] if "source" in m))
    return [{"source": s} for s in sources]


@app.post("/rate")
async def rate_trace(req: RatingRequest):
    from eval.tracer import save_rating

    save_rating(req.trace_id, req.rating)
    return {"status": "ok"}


# ----- Exception handlers -----

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": "Invalid request."})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
