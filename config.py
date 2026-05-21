import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env or environment")

MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
TRACE_DB_PATH = os.getenv("TRACE_DB_PATH", "./data/traces.db")
API_URL = os.getenv("API_URL", "http://localhost:8000")
