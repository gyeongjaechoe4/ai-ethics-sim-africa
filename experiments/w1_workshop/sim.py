"""Headless copy of the simulator rules in app_reinforce.py (Prof. Kang).

Same agent, reward (x10), epsilon schedule, entropy and diversity as the Streamlit app,
without the UI, so the W1 workshop runs can be repeated from the command line.
Any change here must be checked against app_reinforce.py (see verify_professor_table in run_w1.py).
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
from scipy.stats import entropy as scipy_entropy

FRAMEWORKS = ("emotion", "social", "moral", "identity")

# app_reinforce.py get_default_scenarios() rewards, unchanged ("default-v1")
DEFAULT_SCENARIOS: List[dict] = [
    {"sid": "S1", "rewards": {
        "A": {"emotion": 1.0, "social": -0.5, "moral": -1.0, "identity": 0.5},
        "B": {"emotion": -1.0, "social": 0.5, "moral": 1.0, "identity": -0.5}}},
    {"sid": "S2", "rewards": {
        "A": {"emotion": 0.6, "social": -0.8, "moral": -0.7, "identity": 0.3},
        "B": {"emotion": -0.5, "social": 0.9, "moral": 0.6, "identity": 0.4}}},
    {"sid": "S3", "rewards": {
        "A": {"emotion": 0.7, "social": -0.4, "moral": -0.6, "identity": 0.8},
        "B": {"emotion": -0.3, "social": 0.7, "moral": 0.9, "identity": 0.5}}},
]


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(w.values())) or 1.0
    return {k: float(v) / total for k, v in w.items()}


def diversity(actions: List[str]) -> float:
    if not actions:
        return 0.0
    ratio = actions.count("A") / len(actions)
    return float(1.0 - 2.0 * abs(0.5 - ratio))


def run(raw_weights: Dict[str, float], seed: int, episodes: int = 300,
        lr: float = 0.1, epsilon: float = 0.5, decay: float = 0.99, min_eps: float = 0.01,
        scenarios: List[dict] = DEFAULT_SCENARIOS) -> List[dict]:
    """One run = one setting + one seed, from scratch. Returns one row per episode."""
    w = normalize_weights(raw_weights)          # the app normalizes slider values the same way
    random.seed(seed)
    np.random.seed(seed)
    q = {s["sid"]: {"A": 0.0, "B": 0.0} for s in scenarios}
    eps = float(epsilon)
    rows = []
    for ep in range(episodes):
        acts, ep_r = [], 0.0
        for s in scenarios:
            sid = s["sid"]
            if random.random() < eps:
                a = random.choice(["A", "B"])
            elif q[sid]["A"] > q[sid]["B"]:
                a = "A"
            elif q[sid]["B"] > q[sid]["A"]:
                a = "B"
            else:
                a = random.choice(["A", "B"])
            r = sum(float(s["rewards"][a].get(k, 0.0)) * w[k] for k in FRAMEWORKS) * 10.0
            q[sid][a] += lr * (r - q[sid][a])
            acts.append(a)
            ep_r += r
        eps = max(min_eps, eps * decay)
        ents = []
        for v in q.values():
            x = np.array([v["A"], v["B"]], dtype=float)
            x = np.exp(x - x.max())
            ents.append(float(scipy_entropy(x / x.sum())))
        rows.append({"episode": ep + 1, "reward": ep_r, "diversity": diversity(acts),
                     "entropy": float(np.mean(ents)), "actions": "".join(acts), "epsilon_after": eps})
    return rows
