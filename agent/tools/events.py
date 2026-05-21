# agent/tools/events.py
import json
import os
from datetime import datetime

EVENTS_FILE = os.path.join(os.getcwd(), "workspace", "events.json")


def _load_events() -> list:
    if not os.path.exists(EVENTS_FILE):
        return []
    with open(EVENTS_FILE, "r") as f:
        return json.load(f)


def _save_events(events: list) -> None:
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2)


def add_event(date: str, description: str) -> str:
    """Add an event for a specific date (YYYY-MM-DD)."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD."
    events = _load_events()
    events.append({"date": date, "description": description})
    _save_events(events)
    return f"Event added: {date} – {description}"


def list_events(date: str = None) -> str:
    """List all events, optionally filtered by date (YYYY-MM-DD)."""
    events = _load_events()
    if date:
        events = [e for e in events if e["date"] == date]
    if not events:
        return "No events found."
    return "\n".join([f"{e['date']}: {e['description']}" for e in events])
