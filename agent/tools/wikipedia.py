# agent/tools/wikipedia.py
import wikipedia


def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a summary of the top page."""
    try:
        page = wikipedia.page(query, auto_suggest=False)
        return page.summary[:500]  # first 500 characters – concise
    except wikipedia.exceptions.DisambiguationError as e:
        # If multiple pages match, return the options
        options = e.options[:5]
        return f"Multiple matches found. Try one of: {', '.join(options)}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for that query."
    except Exception as e:
        return f"Wikipedia error: {e}"
