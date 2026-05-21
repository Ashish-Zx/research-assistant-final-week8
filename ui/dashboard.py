# ui/dashboard.py
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root is on the path when running via `streamlit run ui/dashboard.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from eval.tracer import add_eval_columns
from config import TRACE_DB_PATH

# Ensure all columns exist before querying
add_eval_columns()


@st.cache_data(ttl=10)
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(TRACE_DB_PATH)
    query = """
        SELECT id, timestamp, user_query, final_answer, steps_json,
               total_duration_ms, goal_completion, efficiency, clarity,
               human_rating
        FROM traces
        ORDER BY timestamp DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df


st.set_page_config(page_title="Agent Dashboard", layout="wide")
st.title("📊 AI Agent – Evaluation Dashboard")

df = load_data()

if df.empty:
    st.warning("No trace data yet. Run some agent queries first.")
    st.stop()

# ------- KPIs -------
today = datetime.now().date()
today_df = df[df["date"] == today]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Queries Today", len(today_df))
col2.metric(
    "Avg Goal Completion",
    f"{today_df['goal_completion'].mean():.2f}" if not today_df.empty else "N/A",
)
col3.metric(
    "Avg Clarity",
    f"{today_df['clarity'].mean():.1f}" if not today_df.empty else "N/A",
)
avg_human = today_df["human_rating"].dropna().mean()
col4.metric(
    "Human Rating (today)",
    f"{avg_human:.1f}/1.0" if not pd.isna(avg_human) else "No ratings",
)

st.divider()

# ------- Tool Usage Chart -------
st.subheader("🔧 Tool Usage (all time)")
tool_counts: dict[str, int] = {}
for steps_json in df["steps_json"]:
    if not steps_json:
        continue
    for step in json.loads(steps_json):
        if step.get("type") == "tool_call":
            tool = step.get("tool", "unknown")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

if tool_counts:
    tool_df = pd.DataFrame(
        {"Tool": list(tool_counts.keys()), "Calls": list(tool_counts.values())}
    ).sort_values("Calls", ascending=True)
    st.bar_chart(tool_df.set_index("Tool"))
else:
    st.caption("No tool calls recorded yet.")

st.divider()

# ------- Success Rate Over Time -------
st.subheader("📈 Goal Completion Rate (by day)")
daily = df.groupby("date")["goal_completion"].mean().reset_index()
daily["date"] = pd.to_datetime(daily["date"])
st.line_chart(daily.set_index("date"))

st.divider()

# ------- Recent Traces Table -------
st.subheader("📋 Recent Traces")
recent = df.head(20)[
    ["timestamp", "user_query", "goal_completion", "efficiency", "clarity", "human_rating"]
]
st.dataframe(recent, use_container_width=True)

# ------- Trace Inspector -------
st.subheader("🔍 Inspect a Trace")
trace_id = st.selectbox("Select a trace ID to view details", df["id"].head(20).tolist())
if trace_id:
    row = df[df["id"] == trace_id].iloc[0]
    st.markdown(f"**User Query:** {row['user_query']}")
    st.markdown(f"**Final Answer:** {row['final_answer']}")
    st.markdown(f"**Duration:** {row['total_duration_ms']} ms")
    st.markdown("**Steps:**")
    steps = json.loads(row["steps_json"]) if row["steps_json"] else []
    for i, step in enumerate(steps):
        st.write(f"{i+1}. {step}")
