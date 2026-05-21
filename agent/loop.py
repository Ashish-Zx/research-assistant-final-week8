# agent/loop.py
import json
import time
import uuid
import threading
import concurrent.futures

from loguru import logger

from llm_client import client, get_groq_model
from config import MAX_AGENT_STEPS
from eval.tracer import save_trace
from eval.judge import evaluate_trace

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to many tools. "
    "Available: calculator, get_weather, web_search, search_wikipedia, "
    "convert_currency, get_news, read_file, write_file, list_files, "
    "add_event, list_events, get_today, search_knowledge_base, summarize_document. "
    "Always use the appropriate tool to answer — do not guess or answer from memory alone. "
    "If you need factual information, try search_knowledge_base (for uploaded documents) "
    "or search_wikipedia (for general knowledge) or web_search (for current events). "
    "For currency conversion, use convert_currency. "
    "For news, use get_news. "
    "When the user asks to summarize, overview, or describe a document or PDF, "
    "ALWAYS call summarize_document immediately — the document is already uploaded, "
    "do NOT ask the user to provide or upload anything. "
    "When you are ready to give the final answer, write 'FINAL ANSWER:' followed by the answer. "
    "Never make up numbers – use exact values from tool results."
)


def trim_messages(messages: list, max_tokens: int = 6000) -> list:
    """Keep system message + last exchanges under max_tokens."""
    if len(messages) <= 1:
        return messages
    total_words = sum(
        len(str(m.get("content", "")) + str(m.get("tool_calls", "")))
        for m in messages
    )
    while total_words > max_tokens * 0.75 and len(messages) > 2:
        messages.pop(1)  # remove oldest non-system message
        total_words = sum(
            len(str(m.get("content", "")) + str(m.get("tool_calls", "")))
            for m in messages
        )
    return messages


def run_agent(
    query: str,
    tools_list: list,
    tools_map: dict,
    user_id: str = "default_user",
    model: str = None,
    max_steps: int = None,
):
    if model is None:
        model = get_groq_model()
    if max_steps is None:
        max_steps = MAX_AGENT_STEPS

    logger.info(f"Agent started: {query} (max_steps={max_steps})")
    trace_id = str(uuid.uuid4())
    yield f"data: {json.dumps({'type': 'trace_id', 'id': trace_id})}\n\n"
    trace_steps = []
    start_time = time.time()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    previous_tool_calls = getattr(run_agent, "previous_tool_calls", [])
    original_user_content = query

    for step in range(max_steps):
        logger.debug(f"Step {step+1}/{max_steps}")

        if step == 0:
            messages[1] = {
                "role": "user",
                "content": original_user_content,
            }
        else:
            messages[1] = {"role": "user", "content": original_user_content}

        try:
            messages = trim_messages(messages)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_list,
                tool_choice="auto",
                timeout=30.0,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            yield f"data: {json.dumps({'type': 'token', 'token': 'I encountered an error.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        msg = resp.choices[0].message

        # --- FINAL ANSWER ---
        if not msg.tool_calls and msg.content:
            content = msg.content
            if "FINAL ANSWER:" in content:
                reasoning, final = content.split("FINAL ANSWER:", 1)
                reasoning = reasoning.strip()
                final = final.strip()
            else:
                reasoning = content
                final = content

            if reasoning:
                yield f"data: {json.dumps({'type': 'thought', 'content': reasoning})}\n\n"
                trace_steps.append({"type": "thought", "content": reasoning})

            for w in final.split():
                yield f"data: {json.dumps({'type': 'token', 'token': w + ' '})}\n\n"

            try:
                from agent.memory import store_memory
                store_memory("default_user", query, final)
                logger.info("Interaction stored in memory.")
            except Exception as e:
                logger.warning(f"Could not store memory: {e}")

            trace_steps.append({"type": "final_answer", "content": final})
            duration_ms = int((time.time() - start_time) * 1000)
            try:
                save_trace(trace_id, query, final, trace_steps, duration_ms)
                logger.info(f"Trace saved: {trace_id} ({duration_ms}ms)")
            except Exception as e:
                logger.warning(f"Could not save trace: {e}")

            def evaluate_trace_async(tid, q, steps, answer):
                try:
                    scores = evaluate_trace(q, steps, answer)
                    from eval.tracer import save_evaluation
                    save_evaluation(
                        tid,
                        scores.get("goal_completion", 0),
                        scores.get("efficiency", 0),
                        scores.get("clarity", 0),
                    )
                    logger.info(f"Evaluation saved for trace {tid}: {scores}")
                except Exception as e:
                    logger.error(f"Evaluation failed for {tid}: {e}")

            threading.Thread(
                target=evaluate_trace_async,
                args=(trace_id, query, trace_steps, final),
                daemon=True,
            ).start()
            break

        # --- TOOL CALLS ---
        if msg.tool_calls:
            yield f"data: {json.dumps({'type': 'thought', 'content': msg.content or 'Deciding to use a tool.'})}\n\n"
            trace_steps.append(
                {"type": "thought", "content": msg.content or "Deciding to use a tool."}
            )

            tool_events = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {"raw": tc.function.arguments}
                tool_events.append({"tool": tc.function.name, "args": args})
                trace_steps.append(
                    {"type": "tool_call", "tool": tc.function.name, "args": args}
                )
            yield f"data: {json.dumps({'type': 'tool_call', 'tools': tool_events})}\n\n"

            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_tc = {}
                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    if func_name in tools_map:
                        try:
                            sig = f"{func_name}:{json.dumps(args, sort_keys=True)}"
                        except Exception:
                            sig = f"{func_name}:{str(args)}"

                        if sig in previous_tool_calls:
                            future = executor.submit(
                                lambda: "Tool already tried – no new information."
                            )
                            logger.info(f"Skipping repeated tool call: {func_name}")
                        else:
                            future = executor.submit(tools_map[func_name], **args)
                            previous_tool_calls.append(sig)
                    else:
                        captured_name = func_name
                        future = executor.submit(
                            lambda n=captured_name: f"Unknown tool: {n}"
                        )
                    future_to_tc[future] = tc

                for future in concurrent.futures.as_completed(
                    future_to_tc, timeout=10
                ):
                    tc = future_to_tc[future]
                    func_name = tc.function.name
                    try:
                        result = future.result()
                        logger.info(f"Tool {func_name} executed successfully")
                    except Exception as e:
                        logger.error(f"Tool {func_name} failed: {e}")
                        result = f"Tool error: {e}"

                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': func_name, 'result': result})}\n\n"
                    trace_steps.append(
                        {"type": "tool_result", "tool": func_name, "result": result}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )

        # --- Empty response ---
        if not msg.tool_calls and not msg.content:
            logger.warning("Empty response from LLM")
            yield f"data: {json.dumps({'type': 'token', 'token': 'I am unable to answer that.'})}\n\n"
            break

    else:
        logger.warning(f"Max steps reached ({max_steps})")
        yield f"data: {json.dumps({'type': 'token', 'token': 'Sorry, I could not complete the task.'})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
    run_agent.previous_tool_calls = previous_tool_calls
