# agent/tools/web.py
from ddgs import DDGS


def web_search(query: str) -> str:
    """Search the web and return the top 3 results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {r['title']}\nLink: {r['href']}\nSnippet: {r['body']}"
            for r in results
        )
    except Exception as e:
        return f"Web search error: {e}"
