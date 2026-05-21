# agent/tools/news.py
from ddgs import DDGS


def get_news(topic: str) -> str:
    """Fetch recent news headlines on a topic."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(topic, max_results=5))
        if not results:
            return "No news found."
        headlines = [f"- {r['title']} ({r.get('date', '')})" for r in results]
        return "\n".join(headlines)
    except Exception as e:
        return f"News error: {e}"
