# agent/guard.py
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Load the model once at module level
_model = joblib.load(Path(__file__).resolve().parents[2] / "failure_predictor.joblib")

def predict_failure_probability(user_query: str) -> float:
    """
    Return the probability (0‑1) that the agent will fail to answer this query.
    Uses the logistic regression model trained on historical traces.
    """
    # Extract features (same as training)
    query_length = len(user_query)
    now = datetime.now()
    hour = now.hour
    day_of_week = now.weekday()  # 0=Monday, 6=Sunday
    # num_tools is unknown before execution; we'll use the median from training
    # For simplicity, we can assume 0 or use a small constant
    num_tools = 0

    # Build a DataFrame with the same column order as during training
    features = pd.DataFrame([{
        'query_length': query_length,
        'num_tools': num_tools,
        'hour': hour,
        'day_of_week': day_of_week
    }])

    # Predict probability of success (class 1), then convert to failure probability
    prob_success = _model.predict_proba(features)[0, 1]
    return float(1 - prob_success)

def should_alert(user_query: str, threshold: float = 0.7) -> bool:
    """Return True if the failure probability exceeds the threshold."""
    return bool(predict_failure_probability(user_query) > threshold)