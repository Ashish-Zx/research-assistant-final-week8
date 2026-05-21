# agent/tools/builtins.py
import requests
from asteval import Interpreter
from loguru import logger

aeval = Interpreter()


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(aeval(expression))
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    url = f"https://wttr.in/{city}?format=%C+%t"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text.strip()
    except Exception as e:
        logger.error(f"Weather error: {e}")
        return f"Weather error: {e}"
