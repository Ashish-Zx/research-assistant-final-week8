# eval/test_runner.py
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from config import API_URL

API_BASE = API_URL


def call_agent(query: str) -> dict:
    """Call the /agent endpoint and collect the full response + trace_id."""
    payload = {"query": query}
    full_answer = ""
    trace_id = None
    tool_calls_seen: list[str] = []

    try:
        with requests.post(
            f"{API_BASE}/agent", json=payload, stream=True, timeout=90
        ) as r:
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
                if etype == "trace_id":
                    trace_id = event.get("id")
                elif etype == "tool_call":
                    for tool in event.get("tools", []):
                        tool_calls_seen.append(tool["tool"])
                elif etype == "token":
                    full_answer += event.get("token", "")
    except Exception as e:
        return {"error": str(e), "trace_id": None, "tools": tool_calls_seen}

    return {
        "answer": full_answer.strip(),
        "trace_id": trace_id,
        "tools": tool_calls_seen,
    }


def evaluate_answer(query: str, tools_used: list[str], answer: str) -> dict:
    """Quick automated evaluation using the LLM judge."""
    from eval.judge import evaluate_trace

    steps = [{"type": "tool_call", "tool": t, "args": {}} for t in tools_used]
    steps.append({"type": "final_answer", "content": answer})
    return evaluate_trace(query, steps, answer)


def run_test_suite(suite_path: str = "tests/test_suite.json") -> bool:
    with open(suite_path) as f:
        suite = json.load(f)

    results = []
    passed = 0
    failed = 0

    for i, test in enumerate(suite, 1):
        query = test["query"]
        print(f"[{i}/{len(suite)}] {query[:80]}...")
        response = call_agent(query)
        if "error" in response:
            print(f"  ERROR: {response['error']}")
            failed += 1
            results.append({**test, "status": "ERROR", "error": response["error"]})
            continue

        scores = evaluate_answer(query, response["tools"], response["answer"])
        goal_ok = scores.get("goal_completion", 0) >= test.get("min_goal_completion", 1)

        if goal_ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        results.append(
            {
                **test,
                "status": status,
                "trace_id": response["trace_id"],
                "tools_used": response["tools"],
                "scores": scores,
            }
        )
        print(
            f"  {status} | goal={scores.get('goal_completion')} "
            f"eff={scores.get('efficiency')} clar={scores.get('clarity')}"
        )
        time.sleep(1)

    print(f"\n{'='*50}")
    print(f"PASSED: {passed}/{len(suite)}")
    print(f"FAILED: {failed}/{len(suite)}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "total": len(suite),
        "passed": passed,
        "failed": failed,
        "details": results,
    }
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report saved to test_report.json")

    return passed == len(suite)


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
