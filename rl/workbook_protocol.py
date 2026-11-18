"""W3 워크북(student_package) 방식 시드 스윕.

app_reinforce.py 의 SimpleEGreedyAgent 와 같은 규칙을 Streamlit 없이 돌린다.
  - seeds 42, 123, 2026, 7777, 31337 / episodes 500 / lr 0.1
  - epsilon 0.5 -> x0.99 매 episode, 최소 0.01
  - 보상 = (시나리오 보상 벡터 · 문화 가중치) x 10, episode 보상 = 시나리오 보상 합
  - entropy = Q 값 softmax 의 Shannon 엔트로피(자연로그) 시나리오 평균
  - diversity = episode 안 A/B 균형 1 - 2|0.5 - A비율|
  - Pearson r = diversity vs reward (500 episode), p-value 포함
가중치는 rl/agent.py 와 같은 규칙(presets/<CODE>.yaml 우선, 없으면 rl/reference_presets.yaml).

실행: python -m rl.workbook_protocol
  -> logs/w3/<CULTURE>_seed<seed>_ep500.csv (35개), logs/w3/summary.json, logs/w3/fig1~3.png
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import entropy as scipy_entropy, f_oneway, pearsonr

from rl.agent import CULTURES, FRAMES, load_weights
from rl.scenario_io import load_normalized

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "w3"
SEEDS = (42, 123, 2026, 7777, 31337)
CFG = dict(episodes=500, lr=0.1, epsilon=0.5, decay=0.99, min_epsilon=0.01, scale=10.0)
TAIL = 100


def run(weights, scenarios, seed, episodes=CFG["episodes"], lr=CFG["lr"], epsilon=CFG["epsilon"],
        decay=CFG["decay"], min_epsilon=CFG["min_epsilon"], scale=CFG["scale"]):
    """app_reinforce.py run_simulation() 과 같은 순서로 난수를 쓴다 (random 모듈)."""
    random.seed(seed)
    np.random.seed(seed)
    q = {s["sid"]: {"A": 0.0, "B": 0.0} for s in scenarios}
    R = {s["sid"]: {a: sum(float(s["rewards"][a][k]) * float(weights.get(k, 0.0)) for k in FRAMES) * scale
                    for a in ("A", "B")} for s in scenarios}
    eps = float(epsilon)
    hist = []
    for ep in range(int(episodes)):
        acts, ep_r, eps_used = [], 0.0, eps
        for s in scenarios:
            sid = s["sid"]
            if random.random() < eps:
                a = random.choice(["A", "B"])
            else:
                qa, qb = q[sid]["A"], q[sid]["B"]
                a = "A" if qa > qb else "B" if qb > qa else random.choice(["A", "B"])
            r = R[sid][a]
            q[sid][a] += lr * (r - q[sid][a])
            acts.append(a)
            ep_r += r
        eps = max(min_epsilon, eps * decay)
        ents = []
        for vals in q.values():
            x = np.array([vals["A"], vals["B"]], float)
            x = np.exp(x - x.max())
            ents.append(float(scipy_entropy(x / x.sum())))
        ratio = acts.count("A") / len(acts)
        hist.append({"episode": ep + 1, "reward": ep_r, "diversity": 1.0 - 2.0 * abs(0.5 - ratio),
                     "entropy": float(np.mean(ents)), "epsilon": eps_used,
                     **{f"act_{s['sid']}": a for s, a in zip(scenarios, acts)}})
    return hist, q


def _ms(xs):
    xs = np.asarray(xs, float)
    return {"mean": round(float(xs.mean()), 4), "sd": round(float(xs.std(ddof=1)), 4) if len(xs) > 1 else 0.0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = load_normalized()
    weights, wsrc = {}, {}
    for c in CULTURES:
        weights[c], wsrc[c] = load_weights(c)

    runs = {}
    for c in CULTURES:
        for sd in SEEDS:
            hist, q = run(weights[c], scenarios, sd)
            runs[(c, sd)] = (hist, q)
            path = OUT / f"{c}_seed{sd}_ep{CFG['episodes']}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(hist[0]))
                w.writeheader()
                for row in hist:
                    w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})

    per_culture, a_rate = {}, {}
    for c in CULTURES:
        rew, ent, div, rs, ps, sig = [], [], [], [], [], 0
        for sd in SEEDS:
            hist, _ = runs[(c, sd)]
            tail = hist[-TAIL:]
            rew.append(np.mean([h["reward"] for h in tail]))
            ent.append(np.mean([h["entropy"] for h in tail]))
            div.append(np.mean([h["diversity"] for h in tail]))
            r, p = pearsonr([h["diversity"] for h in hist], [h["reward"] for h in hist])
            rs.append(float(r))
            ps.append(float(p))
            sig += p < 0.05
        per_culture[c] = {"reward": _ms(rew), "entropy": _ms(ent), "diversity": _ms(div),
                          "pearson_r": _ms(rs), "p_max": float(f"{max(ps):.3g}"), "p_min": float(f"{min(ps):.3g}"),
                          "seeds_p_lt_0.05": int(sig)}
        a_rate[c] = {}
        for s in scenarios:
            col = f"act_{s['sid']}"
            vals = [np.mean([h[col] == "A" for h in runs[(c, sd)][0][-TAIL:]]) for sd in SEEDS]
            a_rate[c][s["sid"]] = _ms(vals)

    # 문화 간 유의차: 5 seed 의 최종 100 ep 보상, 문화 7개 일원분산분석
    groups = [[np.mean([h["reward"] for h in runs[(c, sd)][0][-TAIL:]]) for sd in SEEDS] for c in CULTURES]
    f_stat, f_p = f_oneway(*groups)
    overall_a = float(np.mean([a_rate[c][s["sid"]]["mean"] for c in CULTURES for s in scenarios]))

    # 재설계 참고용 민감도: AFRICA 가중치 한 축을 ±0.05 움직이고 다시 정규화, 200 episode, seed 5개
    base = weights["AFRICA"]
    base_choice = {s["sid"]: ("A" if a_rate["AFRICA"][s["sid"]]["mean"] > 0.5 else "B") for s in scenarios}
    sens = []
    for k in FRAMES:
        for d in (-0.05, 0.05):
            w = dict(base)
            w[k] = max(0.0, w[k] + d)
            tot = sum(w.values())
            w = {kk: v / tot for kk, v in w.items()}
            flips = {}
            for s in scenarios:
                ar = np.mean([np.mean([h[f"act_{s['sid']}"] == "A" for h in run(w, scenarios, sd, episodes=200)[0][-TAIL:]])
                              for sd in SEEDS])
                ch = "A" if ar > 0.5 else "B"
                if ch != base_choice[s["sid"]]:
                    flips[s["sid"]] = f"{base_choice[s['sid']]}->{ch} (A rate {ar:.2f})"
            sens.append({"frame": k, "delta": d, "weights": {kk: round(v, 4) for kk, v in w.items()}, "flips": flips})

    summary = {"protocol": "student_package workbook W3", "config": {**CFG, "seeds": list(SEEDS), "tail": TAIL},
               "weights": weights, "weights_source": wsrc, "scenarios": [s["sid"] for s in scenarios],
               "per_culture": per_culture, "option_A_rate_tail100": a_rate, "overall_option_A_rate": round(overall_a, 4),
               "anova_reward_across_cultures": {"F": round(float(f_stat), 3), "p": float(f"{f_p:.3g}")},
               "sensitivity_AFRICA_pm0.05_200ep": sens}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(runs, scenarios)

    print(f"{len(CULTURES)} cultures x {len(SEEDS)} seeds, {len(scenarios)} scenarios -> {OUT.relative_to(ROOT)}")
    for c, m in per_culture.items():
        print(f"  {c:9s} R {m['reward']['mean']:6.3f}±{m['reward']['sd']:.3f}  H {m['entropy']['mean']:.3f}  "
              f"D {m['diversity']['mean']:.3f}  r {m['pearson_r']['mean']:+.3f}±{m['pearson_r']['sd']:.3f}  "
              f"p≤{m['p_max']:.2g} ({m['seeds_p_lt_0.05']}/5 p<.05)")
    print(f"  overall option A rate {overall_a:.3f} | ANOVA F={f_stat:.2f} p={f_p:.3g}")
    for s in sens:
        if s["flips"]:
            print("  sensitivity", s["frame"], s["delta"], s["flips"])


def make_figures(runs, scenarios):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean_curve(c, key):
        return np.mean([[h[key] for h in runs[(c, sd)][0]] for sd in SEEDS], axis=0)

    ep = np.arange(1, CFG["episodes"] + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in CULTURES:
        y = np.convolve(mean_curve(c, "reward"), np.ones(10) / 10, mode="valid")
        ax.plot(ep[9:], y, label=c, lw=2.2 if c == "AFRICA" else 1.2)
    ax.set(xlabel="Episode", ylabel="Episode reward (5-seed mean, 10-ep moving avg)",
           title="Fig 1. Reward curve by culture preset")
    ax.grid(alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_reward_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in CULTURES:
        ax.plot(ep, mean_curve(c, "entropy"), label=c, lw=2.2 if c == "AFRICA" else 1.2)
    ax.set_yscale("log")
    ax.set(xlabel="Episode", ylabel="Strategy entropy (log scale)", title="Fig 2. Strategy entropy decay")
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_entropy_decay.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, c in zip(axes, ("AFRICA", "EUROPE")):
        x = np.concatenate([[h["diversity"] for h in runs[(c, sd)][0]] for sd in SEEDS])
        y = np.concatenate([[h["reward"] for h in runs[(c, sd)][0]] for sd in SEEDS])
        jitter = np.random.default_rng(0).uniform(-0.02, 0.02, len(x))
        ax.scatter(x + jitter, y, s=6, alpha=0.3)
        z = np.polyfit(x, y, 1)
        xs = np.linspace(0, 1, 50)
        r, p = pearsonr(x, y)
        ax.plot(xs, np.poly1d(z)(xs), "r--", label=f"r={r:.2f}, " + ("p<1e-15" if p < 1e-15 else f"p={p:.1e}"))
        ax.set(xlabel="Diversity (0=one-sided, 1=balanced)", ylabel="Episode reward", title=f"{c} (5 seeds x 500 ep)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Fig 3. Diversity vs reward with regression line")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_diversity_reward.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
