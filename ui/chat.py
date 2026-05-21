# ui/chat.py
import json
import sys
from pathlib import Path

# Ensure the project root is on the path when running via `streamlit run ui/chat.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st
from config import API_URL

st.set_page_config(page_title="AI Research Assistant", page_icon="📚")
st.title("📚 AI Research Assistant")

# ---------- Sidebar ----------
st.sidebar.header("Settings")
BASE_URL = st.sidebar.text_input("API URL", value=API_URL)

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf", "txt", "md"])
if uploaded_file is not None:
    with st.sidebar:
        with st.spinner("Indexing document..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            try:
                resp = requests.post(
                    f"{BASE_URL}/upload", files=files, timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["doc_id"] = data.get("doc_id")
                    st.success("✅ Document ready")
                else:
                    st.error(f"Upload failed: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# ---------- Chat ----------
st.header("Ask the Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                with st.expander(
                    f"🔧 {call['tool']} (click for details)", expanded=False
                ):
                    st.caption(f"**Args:** `{json.dumps(call['args'])}`")
                    st.caption(f"**Result:** {call['result']}")

# ---------- Input ----------
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thought_expander = st.expander("🧠 Agent reasoning", expanded=True)
        thought_placeholder = thought_expander.empty()
        tool_placeholder = st.empty()
        answer_placeholder = st.empty()

        full_answer = ""
        tool_calls_log: list[dict] = []
        current_thought = ""

        payload = {"query": prompt}
        try:
            with requests.post(
                f"{BASE_URL}/agent", json=payload, stream=True, timeout=120
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
                        except Exception:
                            continue

                        etype = event.get("type")

                        if etype == "thought":
                            current_thought += event.get("content", "") + "\n\n"
                            thought_placeholder.markdown(current_thought)

                        elif etype == "trace_id":
                            st.session_state["current_trace_id"] = event.get("id")

                        elif etype == "tool_call":
                            for t in event.get("tools", []):
                                tool_name = t["tool"]
                                args = t["args"]
                                with tool_placeholder.container():
                                    with st.spinner(f"🔧 Calling {tool_name}..."):
                                        st.text(f"Args: {json.dumps(args)}")
                                tool_calls_log.append(
                                    {"tool": tool_name, "args": args, "result": None}
                                )

                        elif etype == "tool_result":
                            tool_name = event.get("tool")
                            result = event.get("result", "")
                            for call in tool_calls_log:
                                if call["tool"] == tool_name and call["result"] is None:
                                    call["result"] = result
                                    break
                            with tool_placeholder.container():
                                st.success(f"✅ {tool_name} completed")
                                st.text(f"Result: {result}")

                        elif etype == "token":
                            full_answer += event.get("token", "")
                            answer_placeholder.markdown(full_answer + "▌")

                        elif etype == "done":
                            answer_placeholder.markdown(full_answer)
                            tool_placeholder.empty()
                            break

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Is it running?")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        if "current_trace_id" in st.session_state:
            trace_id = st.session_state.pop("current_trace_id")
            feedback = st.feedback("thumbs", key=f"fb_{trace_id}")
            if feedback is not None:
                rating = 1 if feedback == 1 else 0
                try:
                    requests.post(
                        f"{BASE_URL}/rate",
                        json={"trace_id": trace_id, "rating": rating},
                        timeout=5,
                    )
                    st.toast("Thanks for your feedback!")
                except Exception:
                    pass

        if full_answer:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                    "tool_calls": tool_calls_log,
                }
            )
