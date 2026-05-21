from openai import OpenAI
from config import GROQ_API_KEY, MODEL_NAME

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

def get_groq_model() -> str:
    return MODEL_NAME