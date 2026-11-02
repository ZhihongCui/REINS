# modules/utils.py

import os
import yaml
import json
import logging
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict

# ---------- Logger Setup ----------
def setup_logger(name="causal_pipeline", log_file="logs/pipeline.log", level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


# ---------- Load YAML Configuration File ----------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------- Save YAML Configuration File ----------
def save_yaml(path: str, content: Dict[str, Any]):
    with open(path, "w") as f:
        yaml.dump(content, f, sort_keys=False)


# ---------- Load JSON ----------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# ---------- Save JSON ----------
def save_json(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------- Normalization ----------
def normalize(x: np.ndarray) -> np.ndarray:
    if np.std(x) == 0:
        return x
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


# ---------- Flatten State Variables ----------
def flatten_state_dict(state: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Flatten nested state variables, for example:
    {"Ego": {"speed": 10}, "Traffic": {"lead_speed": 5}} ->
    {"ego_speed": 10, "traffic_lead_speed": 5}
    """
    flat = {}
    for group, group_vars in state.items():
        for var, val in group_vars.items():
            key = f"{group.lower()}_{var}"
            flat[key] = val
    return flat


# ---------- Timestamp ----------
def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------- Vector Embedding (for rules or conditional expressions) ----------
def string_to_embedding(s: str, dim: int = 32) -> np.ndarray:
    """
    Hash a string into a pseudo-vector, can be used for prototype models, discrete rule embeddings, etc.
    """
    np.random.seed(abs(hash(s)) % (2**32))
    return np.random.rand(dim)


# ---------- LLM Prompt Formatter ----------
def format_prompt(template: str, variables: Dict[str, Any]) -> str:
    return template.format(**variables)


# ---------- Multi-value Grouper ----------
def group_by_type(data: list, key_fn) -> Dict[str, list]:
    groups = defaultdict(list)
    for item in data:
        k = key_fn(item)
        groups[k].append(item)
    return groups

def build_variable_stats(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each numerical variable from sampled episodes.
    Input samples structure: [
      {"trajectory": [
         {"Environment": {...}, "Traffic": {...}, "Ego": {...}, "Action": {...}, "Outcome": {...}, "episode_id": int},
         ...
      ], "seed": int},
      ...
    ]
    Returns: { var_name: {"mean","std","p20","p50","p80","min","max"} }
    """
    acc = defaultdict(list)

    for ep in samples:
        for ts in ep.get("trajectory", []):
            for group, m in ts.items():
                if not isinstance(m, dict):
                    continue
                for k, v in m.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        acc[k].append(float(v))

    stats: Dict[str, Dict[str, float]] = {}
    for k, arr in acc.items():
        if len(arr) < 5:  # Skip if too few samples to avoid noise
            continue
        a = np.asarray(arr, dtype=float)
        stats[k] = {
            "mean": float(np.mean(a)),
            "std":  float(np.std(a) + 1e-8),
            "p20":  float(np.quantile(a, 0.2)),
            "p50":  float(np.quantile(a, 0.5)),
            "p80":  float(np.quantile(a, 0.8)),
            "min":  float(np.min(a)),
            "max":  float(np.max(a)),
        }
    return stats
