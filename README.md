# 📚 AI Research Assistant

A fully autonomous AI agent that reasons step-by-step, uses tools, and explains every decision — with a live evaluation dashboard and automated test suite.

Built with **Groq**, **ChromaDB**, **Streamlit**, and **FastAPI**.

---

## What it does

You ask a question. The agent thinks, picks the right tools, executes them in parallel, and streams its reasoning live to the UI. Every interaction is traced, scored by an LLM judge, and surfaced in a monitoring dashboard.

```
User query → Agent reasons → Calls tools → Streams answer → Saves trace → Evaluates itself
```

---

## Features

**Agent**
- Step-by-step reasoning with visible thought process
- Parallel tool execution with duplicate-call detection
- Automatic context trimming to stay within token limits
- Persistent conversation memory via ChromaDB

**Tools**
| Tool | Description |
|---|---|
| `calculator` | Evaluates math expressions safely |
| `get_weather` | Live weather for any city |
| `web_search` | DuckDuckGo search, top 3 results |
| `search_knowledge_base` | Hybrid vector + BM25 search over uploaded docs |
| `summarize_document` | Summarizes the currently uploaded document |
| `read_file` / `write_file` / `list_files` | Sandboxed file workspace |
| `add_event` / `list_events` | Simple calendar management |
| `get_today` | Returns today's date |

**RAG Pipeline**
- PDF and plain-text ingestion via PyMuPDF
- Chunking with configurable size and overlap
- Hybrid search: dense (sentence-transformers) + sparse (BM25) with RRF fusion
- LLM-based self-query filter extraction (e.g. year metadata)
- Groq-powered reranking

**Evaluation & Monitoring**
- Every trace saved to SQLite (`traces.db`)
- Async LLM-as-judge scoring: goal completion, efficiency, clarity
- Human thumbs up/down feedback in the UI
- Streamlit dashboard with KPIs, charts, and trace inspector

---

## Project structure

```
├── agent.py            # Core agent loop (streaming, tool dispatch, tracing)
├── app.py              # Streamlit chat UI
├── dashboard.py        # Evaluation & monitoring dashboard
├── main.py             # FastAPI backend (upload, agent, rate endpoints)
├── rag.py              # RAG pipeline (ingest, hybrid search, rerank, generate)
├── tools.py            # Tool definitions and registry
├── file_tools.py       # Sandboxed file read/write/list
├── event_tools.py      # Calendar add/list
├── memory.py           # Long-term conversation memory (ChromaDB)
├── tracing.py          # SQLite trace persistence and schema migrations
├── eval_agent.py       # LLM-as-judge evaluation
├── llm_client.py       # Groq API client
├── embedding_client.py # Embedding model client
├── tool_registry.py    # Tool registration and schema generation
├── config.py           # Environment config
├── run_tests.py        # Automated test runner
├── test_suite.json     # 12 test queries covering all tools
└── workspace/          # Sandboxed file storage
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <your-repo-url>
cd research-assistant-final
python -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment**

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=openai/gpt-oss-120b
MAX_AGENT_STEPS=5
LOG_LEVEL=INFO
CHROMA_DB_PATH=./chroma_db
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## Running

**Start the backend**

```bash
python main.py
```

Runs on `http://localhost:8000`.

**Start the chat UI** (in a new terminal)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

**Start the dashboard** (in a new terminal)

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8502`.

---

## Evaluation dashboard

The dashboard auto-refreshes every 10 seconds and shows:

- **KPIs** — queries today, avg goal completion, avg clarity, human rating
- **Tool usage chart** — which tools are called most
- **Goal completion over time** — daily trend line
- **Recent traces table** — last 20 queries with all scores
- **Trace inspector** — click any trace to see every thought, tool call, and result

---

## Automated test suite

`test_suite.json` contains 12 queries covering every tool and common edge cases.

```bash
python run_tests.py
```

Results are written to `test_report.json`. Compare against a saved baseline to catch regressions.

---

## How the agent works

```
1. User sends a query
2. Agent appends reasoning instruction on step 1
3. LLM responds with either:
   a. Tool calls → executed in parallel → results fed back
   b. "FINAL ANSWER:" → reasoning shown in expander, answer streamed to UI
4. Trace saved to SQLite
5. LLM judge scores the trace asynchronously (non-blocking)
6. User can rate the response with 👍 / 👎
```

The agent detects repeated tool calls and skips them to avoid loops. Context is trimmed automatically when the message history grows too large.

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq (configurable model) |
| Embeddings | sentence-transformers |
| Vector store | ChromaDB (persistent) |
| Sparse search | BM25 (rank_bm25) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Tracing | SQLite |
| PDF parsing | PyMuPDF |
| Logging | Loguru |

---

## Notes

- `traces.db` and `chroma_db/` are local data — add them to `.gitignore` or exclude before pushing
- `.env` is never committed
- The `workspace/` directory is sandboxed; file tools cannot escape it
