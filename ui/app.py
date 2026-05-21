# ui/app.py
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import sqlite3

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .thought-block {
        background: rgba(108, 99, 255, 0.12);
        border-left: 4px solid #6c63ff;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        font-size: 0.95rem;
        color: var(--text-color, #e0e0e0);
    }
    .tool-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: var(--text-color, #e0e0e0);
    }
    .final-answer {
        background: rgba(108, 99, 255, 0.08);
        border-top: 3px solid #6c63ff;
        padding: 16px;
        border-radius: 10px;
        margin-top: 16px;
        font-size: 1.05rem;
        color: var(--text-color, #e0e0e0);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Config ----------
BASE_URL = st.secrets.get("API_URL", "http://localhost:8000")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "📊 Analytics"])

# ======================= TAB 1: CHAT =======================
with tab1:
    st.header("Chat with your Agent")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))
            if "tool_calls" in msg:
                for call in msg["tool_calls"]:
                    with st.expander(f"🔧 {call['tool']} (args: {json.dumps(call['args'])})", expanded=False):
                        st.caption(f"Result: {call['result']}")

    # Input box
    if prompt := st.chat_input("Ask anything..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant container
        with st.chat_message("assistant"):
            response_container = st.container()

            full_answer = ""
            tool_calls_log = []
            current_thoughts = []
            trace_id = None

            # Placeholders
            thought_area = response_container.empty()
            tool_area = response_container.empty()
            answer_area = response_container.empty()

            payload = {"query": prompt}
            try:
                with requests.post(
                    f"{BASE_URL}/agent",
                    json=payload,
                    stream=True,
                    timeout=120
                ) as r:
                    if r.status_code != 200:
                        st.error(f"Agent error: {r.text}")
                    else:
                        for line in r.iter_lines(decode_unicode=True):
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                event = json.loads(data_str)
                            except:
                                continue

                            etype = event.get("type")

                            if etype == "trace_id":
                                trace_id = event.get("id")

                            elif etype == "thought":
                                current_thoughts.append(event.get("content", ""))
                                thoughts_html = "".join(
                                    f'<div class="thought-block">💭 {t}</div>' for t in current_thoughts
                                )
                                thought_area.markdown(thoughts_html, unsafe_allow_html=True)

                            elif etype == "tool_call":
                                tools = event.get("tools", [])
                                for t in tools:
                                    tool_calls_log.append({
                                        "tool": t["tool"],
                                        "args": t["args"],
                                        "result": None
                                    })
                                # Render tool cards
                                tool_html = ""
                                for i, call in enumerate(tool_calls_log, 1):
                                    status = "✅" if call["result"] is not None else "⏳"
                                    result_display = f"Result: {call['result']}" if call["result"] else "Waiting..."
                                    tool_html += f"""
                                    <div class="tool-card">
                                        <b>{status} {call['tool']}</b><br>
                                        Args: {json.dumps(call['args'])}<br>
                                        {result_display}
                                    </div>
                                    """
                                tool_area.markdown(tool_html, unsafe_allow_html=True)

                            elif etype == "tool_result":
                                tool_name = event.get("tool")
                                result = event.get("result", "")
                                for call in tool_calls_log:
                                    if call["tool"] == tool_name and call["result"] is None:
                                        call["result"] = result
                                        break
                                # Refresh tool cards
                                tool_html = ""
                                for i, call in enumerate(tool_calls_log, 1):
                                    status = "✅" if call["result"] is not None else "⏳"
                                    result_display = f"Result: {call['result']}" if call["result"] else "Waiting..."
                                    tool_html += f"""
                                    <div class="tool-card">
                                        <b>{status} {call['tool']}</b><br>
                                        Args: {json.dumps(call['args'])}<br>
                                        {result_display}
                                    </div>
                                    """
                                tool_area.markdown(tool_html, unsafe_allow_html=True)

                            elif etype == "token":
                                full_answer += event.get("token", "")
                                answer_area.markdown(f'<div class="final-answer">📝 {full_answer}▌</div>', unsafe_allow_html=True)

                            elif etype == "done":
                                answer_area.markdown(f'<div class="final-answer">✅ {full_answer}</div>', unsafe_allow_html=True)
                                break

            except Exception as e:
                st.error(f"Connection error: {e}")

            # Save message to history
            if full_answer:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "tool_calls": tool_calls_log,
                    "trace_id": trace_id
                })

            # Feedback row
            if trace_id and full_answer:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Helpful", key=f"up_{trace_id}"):
                        requests.post(f"{BASE_URL}/rate", json={"trace_id": trace_id, "rating": 1})
                        st.toast("Thanks for your feedback!")
                with col2:
                    if st.button("👎 Not helpful", key=f"down_{trace_id}"):
                        requests.post(f"{BASE_URL}/rate", json={"trace_id": trace_id, "rating": 0})
                        st.toast("Thanks for your feedback!")

# ======================= TAB 2: DOCUMENTS =======================
with tab2:
    st.header("📁 Document Management")

    # Upload
    uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        if st.button("Index Document"):
            with st.spinner("Indexing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    resp = requests.post(f"{BASE_URL}/upload", files=files, timeout=30)
                    if resp.status_code == 200:
                        st.success(f"✅ Document '{uploaded_file.name}' indexed.")
                    else:
                        st.error(f"Upload failed: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # List uploaded docs
    st.subheader("Indexed Documents")
    if st.button("Refresh List"):
        try:
            resp = requests.get(f"{BASE_URL}/documents")
            if resp.status_code == 200:
                docs = resp.json()
                if docs:
                    df = pd.DataFrame(docs)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No documents indexed yet.")
            else:
                st.warning("Could not fetch documents.")
        except Exception as e:
            st.warning(f"Error: {e}")

# ======================= TAB 3: ANALYTICS =======================
with tab3:
    st.header("📊 Analytics Dashboard")

    # Load traces from SQLite
    @st.cache_data(ttl=10)
    def load_traces():
        try:
            conn = sqlite3.connect("traces.db")
            df = pd.read_sql_query("SELECT * FROM traces", conn)
            conn.close()
            if df.empty:
                return df
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception:
            return pd.DataFrame()

    df = load_traces()

    if df.empty:
        st.info("No traces yet. Start chatting to see analytics.")
    else:
        today = datetime.now().date()
        today_df = df[df["timestamp"].dt.date == today]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Queries Today", len(today_df))
        col2.metric("Avg Goal Completion", f"{today_df['goal_completion'].mean():.2f}" if not today_df.empty else "N/A")
        col3.metric("Avg Clarity", f"{today_df['clarity'].mean():.1f}" if not today_df.empty else "N/A")
        avg_human = today_df["human_rating"].dropna().mean()
        col4.metric("Human Rating (today)", f"{avg_human:.1f}/1.0" if not pd.isna(avg_human) else "No ratings")

        st.divider()

        # Tool usage
        st.subheader("🔧 Tool Usage")
        tool_counts = {}
        for steps_json in df["steps_json"]:
            if not steps_json:
                continue
            steps = json.loads(steps_json)
            for step in steps:
                if step.get("type") == "tool_call":
                    tool = step.get("tool", "unknown")
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1
        if tool_counts:
            tool_df = pd.DataFrame({"Tool": list(tool_counts.keys()), "Calls": list(tool_counts.values())}).sort_values("Calls", ascending=True)
            st.bar_chart(tool_df.set_index("Tool"))
        else:
            st.caption("No tool calls recorded yet.")

        st.divider()

        # Success rate over time
        st.subheader("📈 Goal Completion Rate (by day)")
        if "date" in df.columns:
            daily = df.groupby("date")["goal_completion"].mean().reset_index()
            daily["date"] = pd.to_datetime(daily["date"])
            st.line_chart(daily.set_index("date"))
        else:
            df["date"] = df["timestamp"].dt.date
            daily = df.groupby("date")["goal_completion"].mean().reset_index()
            daily["date"] = pd.to_datetime(daily["date"])
            st.line_chart(daily.set_index("date"))

        st.divider()

        # Recent traces
        st.subheader("📋 Recent Traces")
        recent = df.head(20)[["timestamp", "user_query", "goal_completion", "efficiency", "clarity", "human_rating"]]
        st.dataframe(recent, use_container_width=True)

        # Trace inspector
        st.subheader("🔍 Inspect a Trace")
        trace_id = st.selectbox("Select a trace ID", df["id"].head(20).tolist())
        if trace_id:
            row = df[df["id"] == trace_id].iloc[0]
            st.markdown(f"**User Query:** {row['user_query']}")
            st.markdown(f"**Final Answer:** {row['final_answer']}")
            st.markdown(f"**Duration:** {row['total_duration_ms']} ms")
            if row["steps_json"]:
                steps = json.loads(row["steps_json"])
                for i, step in enumerate(steps):
                    st.write(f"{i+1}. {step}")
