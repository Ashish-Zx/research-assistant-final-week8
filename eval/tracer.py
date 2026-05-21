# eval/tracer.py
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TRACE_DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(TRACE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the traces table and ensure all columns exist."""
    import os
    os.makedirs(os.path.dirname(TRACE_DB_PATH), exist_ok=True)
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS traces (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            user_query TEXT,
            final_answer TEXT,
            steps_json TEXT,
            total_duration_ms INTEGER
        )
        """
    )
    conn.commit()
    conn.close()
    add_eval_columns()


def add_eval_columns() -> None:
    """Add eval/rating columns to the traces table if they don't already exist."""
    columns = [
        ("goal_completion", "INTEGER"),
        ("efficiency", "REAL"),
        ("clarity", "REAL"),
        ("eval_timestamp", "TEXT"),
        ("human_rating", "INTEGER"),
    ]
    conn = get_connection()
    for col_name, col_type in columns:
        try:
            conn.execute(f"ALTER TABLE traces ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


def save_trace(
    trace_id: str,
    user_query: str,
    final_answer: str,
    steps: list,
    duration_ms: int,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO traces
            (id, timestamp, user_query, final_answer, steps_json, total_duration_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            trace_id,
            datetime.now().isoformat(),
            user_query,
            final_answer,
            json.dumps(steps, indent=2),
            duration_ms,
        ),
    )
    conn.commit()
    conn.close()


def save_evaluation(
    trace_id: str,
    goal_completion: int,
    efficiency: float,
    clarity: float,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        UPDATE traces
        SET goal_completion=?, efficiency=?, clarity=?, eval_timestamp=?
        WHERE id=?
        """,
        (goal_completion, efficiency, clarity, datetime.now().isoformat(), trace_id),
    )
    conn.commit()
    conn.close()


def save_rating(trace_id: str, rating: int) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE traces SET human_rating=? WHERE id=?", (rating, trace_id)
    )
    conn.commit()
    conn.close()
