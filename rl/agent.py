"""W3: epsilon-greedy Q-learning (bandit) 학습기 자리.

사양 (강의자료 W3):
  - Q <- Q + alpha * (R - Q),  R = sum_k w_k * r_k
  - episodes=500, seeds=[0..4], alpha=0.10
  - epsilon 1.0 -> 0.05, decay 0.995
  - 7 문화 x 5 seed = 35 runs

W3에서 팀이 구현한다.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ("emotion", "social", "moral", "identity")


def load_preset(culture_code):
    path = ROOT / "presets" / f"{culture_code}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_scenarios():
    path = ROOT / "data" / "scenarios.json"
    return json.loads(path.read_text(encoding="utf-8"))["scenarios"]


def reward(weights, option_reward):
    """R = sum_k w_k * r_k"""
    return sum(weights[k] * option_reward[k] for k in FRAMES)


def train(weights, scenarios, episodes=500, seed=0, alpha=0.10,
          eps_start=1.0, eps_min=0.05, eps_decay=0.995):
    raise NotImplementedError("W3에서 구현")
