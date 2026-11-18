"""W3: epsilon-greedy Q-learning (bandit) 학습기.

사양 (강의자료 W3):
  - 상태 = 시나리오, 행동 = A/B, Q-table = Q[S, A]
  - R = sum_k w_k * r_k,  Q <- Q + alpha * (R - Q)  (bandit, gamma 없음)
  - episodes=500, seeds=[0..4], alpha=0.10, epsilon 1.0 -> 0.05, decay 0.995
  - 7 문화 x 5 seed = 35 runs

한 episode = 모든 시나리오를 한 번씩 결정. epsilon 은 episode 마다 감소한다.

실행:  python -m rl.agent          -> results/runs.csv, results/summary.json 생성
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ("emotion", "social", "moral", "identity")
ACTIONS = ("A", "B")
CULTURES = ("USA", "CHINA", "EUROPE", "KOREA", "LATIN_AM", "MIDDLE_E", "AFRICA")
SEEDS = (0, 1, 2, 3, 4)
CONFIG = dict(episodes=500, alpha=0.10, eps_start=1.0, eps_min=0.05, eps_decay=0.995)
TAIL = 100  # 마지막 100 episode = R[400:500]
RESULTS = ROOT / "results"


# ---------------------------------------------------------------- 입력
def load_preset(culture_code):
    path = ROOT / "presets" / f"{culture_code}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_weights(culture_code):
    """presets/<CODE>.yaml 이 있으면 그 weights, 없으면 강의자료 예시값. (weights, 출처) 반환."""
    path = ROOT / "presets" / f"{culture_code}.yaml"
    if path.exists():
        return {k: float(load_preset(culture_code)["weights"][k]) for k in FRAMES}, f"presets/{culture_code}.yaml"
    ref = yaml.safe_load((ROOT / "rl" / "reference_presets.yaml").read_text(encoding="utf-8"))
    return {k: float(ref["cultures"][culture_code][k]) for k in FRAMES}, "rl/reference_presets.yaml (강의 예시)"


def load_scenarios():
    path = ROOT / "data" / "scenarios.json"
    return json.loads(path.read_text(encoding="utf-8"))["scenarios"]


def reward(weights, option_reward):
    """R = sum_k w_k * r_k"""
    return sum(weights[k] * option_reward[k] for k in FRAMES)


# ---------------------------------------------------------------- 학습
def train(weights, scenarios, episodes=500, seed=0, alpha=0.10,
          eps_start=1.0, eps_min=0.05, eps_decay=0.995):
    """한 문화·한 seed 학습. Q, 선택 기록(episodes x S, 0=A 1=B), 보상 기록, epsilon 기록 반환."""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n = len(scenarios)
    R = np.array([[reward(weights, s["options"][a]["reward"]) for a in ACTIONS] for s in scenarios])
    Q = np.zeros((n, len(ACTIONS)))
    choices = np.zeros((episodes, n), dtype=int)
    rewards = np.zeros((episodes, n))
    eps_trace = np.zeros(episodes)
    eps = eps_start
    for ep in range(episodes):
        eps_trace[ep] = eps
        for s in range(n):
            if rng.random() < eps:
                a = int(rng.integers(len(ACTIONS)))
            else:
                a = int(np.argmax(Q[s]))  # 동률이면 A
            r = R[s, a]
            Q[s, a] += alpha * (r - Q[s, a])
            choices[ep, s] = a
            rewards[ep, s] = r
        eps = max(eps_min, eps * eps_decay)
    return {"Q": Q, "choices": choices, "rewards": rewards, "eps": eps_trace}


def entropy2(p):
    """H = -sum p log2 p (A/B 두 옵션)"""
    return -sum(x * math.log2(x) for x in (p, 1 - p) if x > 0)


def run_one(culture, seed, weights, scenarios, cfg=CONFIG):
    out = train(weights, scenarios, seed=seed, **cfg)
    tail_c = out["choices"][-TAIL:]
    tail_r = out["rewards"][-TAIL:]
    rows = []
    for i, s in enumerate(scenarios):
        p_b = float(tail_c[:, i].mean())
        rows.append({
            "culture": culture, "seed": seed, "scenario": s["id"],
            "reward_mean": round(float(tail_r[:, i].mean()), 6),
            "p_A": round(1 - p_b, 6), "p_B": round(p_b, 6),
            "entropy": round(entropy2(p_b), 6),
            "greedy": ACTIONS[int(np.argmax(out["Q"][i]))],
            "Q_A": round(float(out["Q"][i, 0]), 6), "Q_B": round(float(out["Q"][i, 1]), 6),
        })
    return rows


def run_all(cultures=CULTURES, seeds=SEEDS):
    scenarios = load_scenarios()
    weights, wsrc = {}, {}
    for c in cultures:
        weights[c], wsrc[c] = load_weights(c)
    rows = [r for c in cultures for sd in seeds for r in run_one(c, sd, weights[c], scenarios)]
    return rows, weights, wsrc, scenarios


# ---------------------------------------------------------------- 지표
def _ms(xs):
    xs = np.asarray(xs, dtype=float)
    return {"mean": round(float(xs.mean()), 4), "sd": round(float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, 4)}


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None  # 한쪽이 상수면 정의 안 됨
    return round(float(np.corrcoef(x, y)[0, 1]), 4)


def summarize(rows, weights, scenarios):
    cultures = list(weights)
    by = {}
    for r in rows:
        by.setdefault((r["culture"], r["scenario"]), []).append(r)

    # 문화 x 시나리오 (seed 평균 ± SD)
    cells = {}
    for (c, sid), rs in by.items():
        greedy = [r["greedy"] for r in rs]
        cells.setdefault(c, {})[sid] = {
            "reward": _ms([r["reward_mean"] for r in rs]),
            "p_A": _ms([r["p_A"] for r in rs]),
            "entropy": _ms([r["entropy"] for r in rs]),
            "choice": max(ACTIONS, key=greedy.count),
            "choice_agreement": greedy.count(max(ACTIONS, key=greedy.count)) / len(greedy),
        }

    # 문화별: 시나리오 평균 보상의 seed 간 평균 ± SD
    per_culture = {}
    for c in cultures:
        seeds = sorted({r["seed"] for r in rows if r["culture"] == c})
        rew = [np.mean([r["reward_mean"] for r in rows if r["culture"] == c and r["seed"] == s]) for s in seeds]
        ent = [np.mean([r["entropy"] for r in rows if r["culture"] == c and r["seed"] == s]) for s in seeds]
        pa = [np.mean([r["p_A"] for r in rows if r["culture"] == c and r["seed"] == s]) for s in seeds]
        per_culture[c] = {"reward": _ms(rew), "entropy": _ms(ent), "p_A": _ms(pa),
                          "share_A_choices": sum(cells[c][s["id"]]["choice"] == "A" for s in scenarios) / len(scenarios)}

    # Diversity: 시나리오별 7개 문화의 (다수결) 선택 종류 수 / 7
    diversity = {}
    for s in scenarios:
        ch = [cells[c][s["id"]]["choice"] for c in cultures]
        diversity[s["id"]] = {"D": round(len(set(ch)) / len(cultures), 4),
                              "choices": dict(zip(cultures, ch)),
                              "minority_share": round(min(ch.count("A"), ch.count("B")) / len(cultures), 4)}

    # Pearson r(w_frame, p_choice): 프레임 k 보상이 더 큰 옵션을 고를 확률 vs w_k (7문화)
    pearson = {"per_scenario": {}, "pooled": {}}
    pooled = {k: ([], []) for k in FRAMES}
    for s in scenarios:
        pearson["per_scenario"][s["id"]] = {}
        for k in FRAMES:
            rA, rB = s["options"]["A"]["reward"][k], s["options"]["B"]["reward"][k]
            if rA == rB:
                continue
            fav = "A" if rA > rB else "B"
            w = [weights[c][k] for c in cultures]
            p = [cells[c][s["id"]]["p_A"]["mean"] if fav == "A" else 1 - cells[c][s["id"]]["p_A"]["mean"] for c in cultures]
            pearson["per_scenario"][s["id"]][k] = {"favored": fav, "r": _pearson(w, p)}
            pooled[k][0].extend(w)
            pooled[k][1].extend(p)
    for k in FRAMES:
        pearson["pooled"][k] = {"r": _pearson(*pooled[k]), "n": len(pooled[k][0])}

    return {"cells": cells, "per_culture": per_culture, "diversity": diversity, "pearson": pearson}


def write_results(rows, weights, wsrc, scenarios, out_dir=RESULTS):
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"config": {**CONFIG, "seeds": list(SEEDS), "tail": TAIL,
                          "eps_final": round(max(CONFIG["eps_min"], CONFIG["eps_decay"] ** (CONFIG["episodes"] - 1)), 4)},
               "weights": weights, "weights_source": wsrc,
               "scenarios": [s["id"] for s in scenarios],
               **summarize(rows, weights, scenarios)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    rows, weights, wsrc, scenarios = run_all()
    s = write_results(rows, weights, wsrc, scenarios)
    print(f"{len(weights)} cultures x {len(SEEDS)} seeds = {len(weights) * len(SEEDS)} runs, {len(scenarios)} scenarios")
    for c, m in s["per_culture"].items():
        print(f"  {c:9s} reward {m['reward']['mean']:+.3f}±{m['reward']['sd']:.3f}  "
              f"H {m['entropy']['mean']:.3f}±{m['entropy']['sd']:.3f}  A선택비율 {m['share_A_choices']:.2f}  [{wsrc[c]}]")
    for sid, d in s["diversity"].items():
        print(f"  D[{sid}] = {d['D']:.3f}  {d['choices']}")
    print("  Pearson pooled:", {k: v["r"] for k, v in s["pearson"]["pooled"].items()})


if __name__ == "__main__":
    main()
