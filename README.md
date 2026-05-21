# 📚 AI Research Assistant

A full-stack, agentic AI research assistant that combines a **ReAct-style tool-calling agent**, a **Retrieval-Augmented Generation (RAG) pipeline**, and a **real-time Streamlit UI** — all powered by Groq's ultra-fast LLM inference.

🚀 **Live Demo:** [research-assistant-final.streamlit.app](https://research-assistant-final.streamlit.app)

---

## ✨ Features

### 🤖 Agentic Reasoning
- Multi-step **ReAct agent loop** — the model reasons, picks tools, observes results, and iterates until it reaches a final answer
- Parallel tool execution via `ThreadPoolExecutor` for faster multi-tool queries
- Duplicate tool-call detection to avoid redundant API calls
- Automatic context trimming to stay within token limits
- Streaming responses via **Server-Sent Events (SSE)** for real-time token-by-token output

### 🛠️ Tool Suite (14 tools)

| Tool | Description |
|---|---|
| `calculator` | Evaluates mathematical expressions safely via `asteval` |
| `get_weather` | Fetches live weather for any city via wttr.in |
| `web_search` | DuckDuckGo web search — top 3 results |
| `get_news` | Recent news headlines on any topic via DuckDuckGo News |
| `search_wikipedia` | Wikipedia page summaries |
| `convert_currency` | Live exchange rates via exchangerate-api.com |
| `search_knowledge_base` | Semantic + BM25 hybrid search over uploaded documents |
| `summarize_document` | Auto-summarizes the most recently uploaded document |
| `read_file` | Reads files from a sandboxed workspace directory |
| `write_file` | Writes files to the sandboxed workspace |
| `list_files` | Lists all files in the workspace |
| `add_event` | Adds a dated event to a local JSON calendar |
| `list_events` | Lists calendar events, optionally filtered by date |
| `get_today` | Returns today's date |

### 📄 RAG Pipeline
- **Document ingestion:** PDF and plain-text/Markdown files via PyMuPDF
- **Chunking:** Overlapping word-level chunks (200 words, 50-word overlap)
- **Embeddings:** Local `all-MiniLM-L6-v2` (384-dim) via `sentence-transformers`, or Ollama `nomic-embed-text` (768-dim)
- **Vector store:** ChromaDB with persistent storage
- **Hybrid search:** Vector similarity + BM25 (Okapi) fused via **Reciprocal Rank Fusion (RRF)**
- **Self-query filtering:** LLM extracts metadata filters (e.g., year) from natural language queries
- **LLM re-ranking:** Top-20 candidates re-ranked to top-5 by the LLM before answer generation

### 🧠 Conversation Memory
- Every interaction is summarized by the LLM and stored as a vector embedding in ChromaDB
- Relevant past interactions are retrieved and injected into the agent's context on each new query

### 📊 Evaluation & Observability
- Every agent run is traced and persisted to **SQLite** (`traces.db`)
- Async **LLM-as-judge** evaluation scores each trace on:
  - `goal_completion` (0/1)
  - `efficiency` (1–5)
  - `clarity` (1–5)
- Human feedback (👍/👎) collected in-UI and stored per trace
- Full **Analytics Dashboard** with KPIs, tool usage charts, daily goal completion trends, and a trace inspector

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (ui/)                    │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   Chat   │  │  Documents   │  │  Analytics (dash) │  │
│  └────┬─────┘  └──────┬───────┘  └───────────────────┘  │
└───────┼───────────────┼─────────────────────────────────┘
        │ SSE stream    │ file upload
┌───────▼───────────────▼─────────────────────────────────┐
│                  FastAPI Backend (api/)                  │
│   /agent (SSE)   /upload   /ask   /documents   /rate    │
└───────┬─────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────┐
│                  Agent Loop (agent/loop.py)               │
│  System Prompt → LLM (Groq) → Tool Calls → Observations  │
│  ↕ parallel ThreadPoolExecutor   ↕ SSE yield             │
└───────┬──────────────────────────────────────────────────┘
        │
   ┌────┴──────────────────────────────────────────┐
   │              Tool Registry (14 tools)          │
   │  builtins · web · wikipedia · news · currency  │
   │  files · events · RAG (search + summarize)     │
   └────┬──────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────┐
│                    RAG Pipeline (rag/)                    │
│  chunker → embedder → ChromaDB + BM25 → RRF → rerank    │
└──────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────┐
│              Eval & Memory (eval/ · agent/memory.py)      │
│  SQLite traces · LLM judge · human ratings · ChromaDB    │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com/)

### 1. Clone & install

```bash
git clone https://github.com/your-username/research-assistant-final.git
cd research-assistant-final
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=meta-llama/llama-4-scout-17b-16e-instruct
MAX_AGENT_STEPS=5
LOG_LEVEL=INFO
CHROMA_DB_PATH=./data/chroma_db
TRACE_DB_PATH=./data/traces.db
API_URL=http://localhost:8000
EMBEDDING_PROVIDER=transformers   # or "ollama"
```

### 3. Start the backend

```bash
python run_api.py
# FastAPI running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Start the UI

```bash
# Full-featured app (chat + documents + analytics)
streamlit run ui/app.py

# Or the standalone chat UI
python run_chat.py

# Or the standalone analytics dashboard
python run_dashboard.py
```

---

## 🐳 Docker

```bash
docker build -t research-assistant .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key research-assistant
```

The image pre-downloads `all-MiniLM-L6-v2` at build time so the first request is instant. Persistent data (ChromaDB, traces, model cache) is stored under `/data` — mount a volume to preserve it across restarts:

```bash
docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_key \
  -v $(pwd)/data:/data \
  research-assistant
```

---

## 📁 Project Structure

```
research-assistant-final/
├── agent/
│   ├── loop.py            # ReAct agent loop (SSE streaming, parallel tools)
│   ├── memory.py          # ChromaDB-backed conversation memory
│   └── tools/
│       ├── registry.py    # Tool class + ToolRegistry (auto JSON Schema)
│       ├── builtins.py    # calculator, get_weather
│       ├── web.py         # web_search (DuckDuckGo)
│       ├── news.py        # get_news (DuckDuckGo News)
│       ├── wikipedia.py   # search_wikipedia
│       ├── currency.py    # convert_currency
│       ├── files.py       # read_file, write_file, list_files
│       └── events.py      # add_event, list_events
├── api/
│   ├── main.py            # FastAPI app — /agent, /upload, /ask, /rate, /documents
│   └── models.py          # Pydantic request models
├── rag/
│   ├── chunker.py         # PDF/text extraction + overlapping chunking
│   ├── embedder.py        # sentence-transformers or Ollama embeddings
│   ├── database.py        # ChromaDB ingestion + BM25 index builder
│   └── retriever.py       # Hybrid search (RRF), LLM rerank, answer gen, summarize
├── eval/
│   ├── tracer.py          # SQLite trace persistence
│   ├── judge.py           # LLM-as-judge evaluation (goal/efficiency/clarity)
│   └── test_runner.py     # Automated test suite runner
├── ui/
│   ├── app.py             # Full Streamlit app (chat + docs + analytics tabs)
│   ├── chat.py            # Standalone chat UI
│   └── dashboard.py       # Standalone analytics dashboard
├── config.py              # Centralised config from .env
├── llm_client.py          # Groq OpenAI-compatible client
├── Dockerfile             # Production Docker image
├── requirements.txt
└── .env.example
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Your Groq API key |
| `MODEL_NAME` | `meta-llama/llama-4-scout-17b-16e-instruct` | Any Groq-hosted model |
| `MAX_AGENT_STEPS` | `5` | Max reasoning iterations per query |
| `LOG_LEVEL` | `INFO` | Loguru log level |
| `CHROMA_DB_PATH` | `./data/chroma_db` | ChromaDB persistence directory |
| `TRACE_DB_PATH` | `./data/traces.db` | SQLite traces database path |
| `API_URL` | `http://localhost:8000` | Backend URL used by the Streamlit UI |
| `EMBEDDING_PROVIDER` | `transformers` | `transformers` (local) or `ollama` |

> **Note:** Switching `EMBEDDING_PROVIDER` requires deleting `chroma_db/` to avoid vector dimension mismatches.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/agent` | Run the agent (SSE stream) |
| `POST` | `/upload` | Upload and index a PDF/text document |
| `POST` | `/ask` | Direct RAG query (no agent loop) |
| `POST` | `/rate` | Submit human feedback for a trace |
| `GET` | `/documents` | List indexed document sources |
| `GET` | `/health` | Health check |

**SSE event types** emitted by `/agent`:

| Type | Payload | Description |
|---|---|---|
| `trace_id` | `{ id }` | Unique ID for this run |
| `thought` | `{ content }` | Agent's internal reasoning |
| `tool_call` | `{ tools: [{tool, args}] }` | Tools about to be called |
| `tool_result` | `{ tool, result }` | Result from a tool |
| `token` | `{ token }` | Streamed answer token |
| `done` | — | Stream complete |

---

## 🧪 Running the Test Suite

```bash
# Make sure the API is running first
python run_api.py &

# Run the automated eval suite
python eval/test_runner.py
# Results saved to test_report.json
```

---

## 🛠️ Extending the Agent

Adding a new tool takes three steps:

1. **Write the function** in `agent/tools/your_tool.py`
2. **Register it** in `api/main.py`:
   ```python
   tool_registry.register(Tool("my_tool", "Description of what it does.", my_function))
   ```
3. **Mention it** in the `SYSTEM_PROMPT` in `agent/loop.py` so the agent knows it exists

The `Tool` class in `registry.py` automatically generates the OpenAI-compatible JSON Schema from the function's type hints — no manual schema writing needed.

---

## 🤝 Contributing

1. Fork the repo and create a feature branch
2. Make your changes and add tests to `tests/test_suite.json`
3. Run `python eval/test_runner.py` and confirm all tests pass
4. Open a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
