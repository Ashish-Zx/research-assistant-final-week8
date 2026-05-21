# eval/judge.py
import json
import re
import sys
from pathlib import Path

# Ensure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import client, get_groq_model


def evaluate_trace(user_query: str, steps: list, final_answer: str) -> dict:
    """Return a dict with goal_completion, efficiency, and clarity scores."""
    steps_str = json.dumps(steps, indent=2)
    prompt = f"""You are an impartial evaluator assessing an AI agent's performance.

User query: {user_query}

Agent steps (thoughts, tool calls, results):
{steps_str}

Agent final answer: {final_answer}

Rate the interaction on these scales:
- goal_completion: 0 if the agent failed to answer the user's request, 1 if it succeeded.
- efficiency: 1 = extremely wasteful, unnecessary tool calls; 5 = perfect number of steps.
- clarity: 1 = confusing or poorly written final answer; 5 = extremely clear and concise.

Return a JSON object with keys: goal_completion, efficiency, clarity. Only JSON, no other text.
Example: {{"goal_completion": 1, "efficiency": 4, "clarity": 5}}"""

    response = client.chat.completions.create(
        model=get_groq_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    pattern = r"(goal_completion|efficiency|clarity)\s*[:=]\s*([\d.]+)"
    scores: dict = {}
    for m in re.findall(pattern, raw, re.IGNORECASE):
        key = m[0].lower()
        val = float(m[1])
        scores[key] = int(val) if key == "goal_completion" else val
    return (
        scores
        if len(scores) == 3
        else {"goal_completion": 0, "efficiency": 0, "clarity": 0}
    )
