"""W1 workshop (11주차): baseline run, exact replication, one-factor paired-seed comparison.

Usage (repo root):  python experiments/w1_workshop/run_w1.py
Outputs go to experiments/w1_workshop/out/.  All settings are fixed below BEFORE running.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim import FRAMEWORKS, normalize_weights, run  # noqa: E402

OUT = HERE / "out"
RUNS = OUT / "runs"

# ---------------- pre-registered plan (do not edit after seeing results) ----------------
# Weights = team's "AFRICA 최적설정" (2026-11-18): best point inside the AFRICA evidence band
# S .30-.40, I .20-.30, E .15-.30, M .15-.25 for the 3 default scenarios.
AFRICA_OPT = {"emotion": 0.15, "social": 0.40, "moral": 0.25, "identity": 0.20}
KOREA_WS = {"emotion": 0.2, "social": 0.2, "moral": 0.2, "identity": 0.4}  # workshop slide 23 baseline
LEARN = {"lr": 0.1, "epsilon": 0.5, "decay": 0.99, "min_eps": 0.01}
EPISODES = 300
BASE_SEED = 42
SEEDS = [42, 123, 2026, 7777, 31337]           # 5 seeds fixed before the run, shared by both conditions
WINDOW = (251, 300)                              # evaluation window (episodes, inclusive)
FACTOR = "identity"
LEVELS = {"A": 0.20, "B": 0.30}                  # raw identity value; the other raw values stay as AFRICA_OPT
SENS_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]  # sensitivity analysis (extra)
SCENARIO_VERSION = "default-v1 (app_reinforce.py get_default_scenarios)"
TOL = 1e-9                                       # same-environment replication tolerance
# ---------------------------------------------------------------------------------------


def commit_id() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def save_run(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.10f}" if isinstance(v, float) else v) for k, v in r.items()})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def window_stats(rows, key):
    lo, hi = WINDOW
    xs = [r[key] for r in rows if lo <= r["episode"] <= hi]
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


def a_ratio(rows):
    lo, hi = WINDOW
    acts = "".join(r["actions"] for r in rows if lo <= r["episode"] <= hi)
    return acts.count("A") / len(acts)


def sign_test_one_sided(up: int, down: int) -> float:
    n = up + down
    if n == 0:
        return float("nan")
    return sum(comb(n, k) for k in range(up, n + 1)) / 2 ** n


def greedy(w) -> str:
    """Best option per scenario under weights w; '=' marks an exact tie (agent then picks at random)."""
    from sim import DEFAULT_SCENARIOS
    out = ""
    for s in DEFAULT_SCENARIOS:
        ga = sum(s["rewards"]["A"][f] * w[f] for f in FRAMEWORKS)
        gb = sum(s["rewards"]["B"][f] * w[f] for f in FRAMEWORKS)
        out += "=" if abs(ga - gb) < 1e-9 else ("A" if ga > gb else "B")
    return out


def weights_for(level: float):
    raw = dict(AFRICA_OPT)
    raw[FACTOR] = level
    return raw


def main():
    OUT.mkdir(exist_ok=True)
    log = []  # experiment log (slide 38 fields)

    def record(run_id, cond, seed, raw, rows, path):
        w = normalize_weights(raw)
        d_m, d_sd = window_stats(rows, "diversity")
        r_m, r_sd = window_stats(rows, "reward")
        h_m, h_sd = window_stats(rows, "entropy")
        row = {
            "run_id": run_id, "condition": cond, "seed": seed,
            "raw_weights": "/".join(f"{raw[k]:.2f}" for k in FRAMEWORKS),
            "weights_normalized": "/".join(f"{w[k]:.3f}" for k in FRAMEWORKS),
            "alpha_eps_decay_min": f"{LEARN['lr']}/{LEARN['epsilon']}/{LEARN['decay']}/{LEARN['min_eps']}",
            "episodes": EPISODES, "scenario": "default-v1", "commit": commit_id(),
            "window": f"{WINDOW[0]}~{WINDOW[1]}", "output": str(path.relative_to(HERE)), "sha256_12": sha(path),
            "diversity_mean": round(d_m, 4), "diversity_sd": round(d_sd, 4),
            "reward_mean": round(r_m, 4), "reward_sd": round(r_sd, 4),
            "entropy_mean": round(h_m, 4), "A_ratio": round(a_ratio(rows), 4),
            "final_greedy_choice": "".join(rows[-1]["actions"]),
        }
        log.append(row)
        return row

    # 1) baseline run + 2) exact replication (run twice, compare row by row)
    base_raw = AFRICA_OPT
    b1 = run(base_raw, BASE_SEED, EPISODES, **LEARN)
    p1 = RUNS / f"BASE_AFRICAopt_seed{BASE_SEED}_run1.csv"
    save_run(b1, p1)
    b2 = run(base_raw, BASE_SEED, EPISODES, **LEARN)
    p2 = RUNS / f"BASE_AFRICAopt_seed{BASE_SEED}_run2.csv"
    save_run(b2, p2)
    record("BASE_AFRICAopt_seed42_run1", "baseline", BASE_SEED, base_raw, b1, p1)
    record("BASE_AFRICAopt_seed42_run2", "baseline-replication", BASE_SEED, base_raw, b2, p2)
    max_diff = max(abs(x[k] - y[k]) for x, y in zip(b1, b2) for k in ("reward", "diversity", "entropy"))
    replication = {
        "rows_run1": len(b1), "rows_run2": len(b2),
        "max_abs_diff": max_diff, "tolerance": TOL,
        "sha_run1": sha(p1), "sha_run2": sha(p2),
        "identical_files": sha(p1) == sha(p2),
        "pass": len(b1) == len(b2) and max_diff <= TOL,
    }

    # workshop reference baseline (KOREA, slide 23) for orientation
    k = run(KOREA_WS, BASE_SEED, EPISODES, **LEARN)
    pk = RUNS / "REF_KOREA_seed42.csv"
    save_run(k, pk)
    record("REF_KOREA_seed42", "reference(KOREA slide 23)", BASE_SEED, KOREA_WS, k, pk)

    # 3) one-factor paired-seed comparison
    pairs = []
    for sd in SEEDS:
        res = {}
        for cond, lvl in LEVELS.items():
            raw = weights_for(lvl)
            rows = run(raw, sd, EPISODES, **LEARN)
            rid = f"{cond}_{FACTOR}{int(round(lvl * 100))}_seed{sd}"
            path = RUNS / f"{rid}.csv"
            save_run(rows, path)
            res[cond] = record(rid, cond, sd, raw, rows, path)
        delta = {m: res["B"][m] - res["A"][m] for m in ("diversity_mean", "reward_mean", "entropy_mean", "A_ratio")}
        d = delta["diversity_mean"]
        pairs.append({"seed": sd, "A_div": res["A"]["diversity_mean"], "B_div": res["B"]["diversity_mean"],
                      "delta_div": round(d, 4), "delta_reward": round(delta["reward_mean"], 4),
                      "delta_entropy": round(delta["entropy_mean"], 4), "delta_A_ratio": round(delta["A_ratio"], 4),
                      "direction": "상승" if d > 1e-12 else ("하락" if d < -1e-12 else "동점")})
    deltas = np.array([p["delta_div"] for p in pairs])
    up = sum(p["direction"] == "상승" for p in pairs)
    down = sum(p["direction"] == "하락" for p in pairs)
    tie = sum(p["direction"] == "동점" for p in pairs)
    comparison = {
        "factor": FACTOR, "levels_raw": LEVELS,
        "weights_A": normalize_weights(weights_for(LEVELS["A"])),
        "weights_B": normalize_weights(weights_for(LEVELS["B"])),
        "planned_pairs": len(SEEDS), "completed": len(pairs), "missing": 0,
        "up": up, "tie": tie, "down": down,
        "mean_delta_div": float(deltas.mean()), "sd_delta_div": float(deltas.std(ddof=1)),
        "mean_delta_reward": float(np.mean([p["delta_reward"] for p in pairs])),
        "sign_test_p_lower_div(B<A)": sign_test_one_sided(down, up),
    }

    # 4) sensitivity analysis (extra, slide 34)
    sens = []
    for lvl in SENS_LEVELS:
        raw = weights_for(lvl)
        ds, rs, ar = [], [], []
        for sd in SEEDS:
            rows = run(raw, sd, EPISODES, **LEARN)
            ds.append(window_stats(rows, "diversity")[0]); rs.append(window_stats(rows, "reward")[0]); ar.append(a_ratio(rows))
        w = normalize_weights(raw)
        sens.append({"identity_raw": lvl, "identity_norm": round(w["identity"], 3),
                     "weights_norm": "/".join(f"{w[k]:.3f}" for k in FRAMEWORKS),
                     "diversity_mean": round(float(np.mean(ds)), 4), "diversity_sd": round(float(np.std(ds, ddof=1)), 4),
                     "reward_mean": round(float(np.mean(rs)), 3), "A_ratio": round(float(np.mean(ar)), 4),
                     "greedy_choice": greedy(w)})

    # write tables
    for name, rows in (("experiment_log.csv", log), ("paired_comparison.csv", pairs), ("sensitivity_identity.csv", sens)):
        with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
    env = {"python": platform.python_version(), "numpy": np.__version__,
           "scipy": __import__("scipy").__version__, "commit": commit_id(), "platform": platform.platform()}
    summary = {"plan": {"baseline_weights_raw": AFRICA_OPT, "learn": LEARN, "episodes": EPISODES,
                        "seed_baseline": BASE_SEED, "seeds": SEEDS, "window": WINDOW, "factor": FACTOR,
                        "levels": LEVELS, "scenario": SCENARIO_VERSION},
               "env": env, "replication": replication, "comparison": comparison,
               "baseline_seed42": {k_: v for k_, v in log[0].items() if k_.endswith(("mean", "sd", "ratio", "choice"))},
               "korea_ref_seed42": {k_: v for k_, v in log[2].items() if k_.endswith(("mean", "sd", "ratio", "choice"))}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) intro deck: reproduce Prof. Kang's KOREA 100 vs 300 table, then the same test for AFRICA opt
    prof = {597495: (7.56, 9.20), 529552: (8.80, 8.40), 249614: (9.20, 9.20), 656981: (7.96, 9.20),
            403401: (8.40, 9.20), 700833: (9.20, 9.20), 647191: (9.20, 9.20), 38067: (8.80, 9.20),
            65535: (6.72, 9.20), 658613: (8.36, 9.20), 405631: (8.36, 9.20), 843340: (8.80, 9.20)}
    chk = []
    for sd, (a100, a300) in prof.items():
        m100 = float(np.mean([r["reward"] for r in run(KOREA_WS, sd, 100, **LEARN)[-5:]]))
        m300 = float(np.mean([r["reward"] for r in run(KOREA_WS, sd, 300, **LEARN)[-5:]]))
        chk.append({"seed": sd, "prof_100": a100, "ours_100": round(m100, 2), "prof_300": a300, "ours_300": round(m300, 2),
                    "match": round(m100, 2) == a100 and round(m300, 2) == a300})
    rng = np.random.default_rng(20261123)          # seeds drawn before looking at any AFRICA 100/300 result
    new_seeds = [int(x) for x in rng.choice(np.arange(1, 1_000_001), size=12, replace=False)]
    ep_rows = []
    for sd in new_seeds:
        m100 = float(np.mean([r["reward"] for r in run(AFRICA_OPT, sd, 100, **LEARN)[-5:]]))
        m300 = float(np.mean([r["reward"] for r in run(AFRICA_OPT, sd, 300, **LEARN)[-5:]]))
        d = m300 - m100
        ep_rows.append({"seed": sd, "ep100_last5": round(m100, 2), "ep300_last5": round(m300, 2), "delta": round(d, 2),
                        "direction": "상승" if d > 1e-9 else ("하락" if d < -1e-9 else "동점")})
    u = sum(r["direction"] == "상승" for r in ep_rows); dn = sum(r["direction"] == "하락" for r in ep_rows)
    summary["intro_professor_table_check"] = {"pairs": len(chk), "all_match": all(c["match"] for c in chk)}
    summary["intro_africa_100_vs_300"] = {"seeds": new_seeds, "up": u, "down": dn, "tie": len(ep_rows) - u - dn,
                                          "mean_delta": round(float(np.mean([r["delta"] for r in ep_rows])), 3),
                                          "sign_test_p_one_sided": sign_test_one_sided(u, dn)}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in (("intro_professor_table_check.csv", chk), ("intro_africa_100_vs_300.csv", ep_rows)):
        with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    for r in chk: print(r)
    for r in ep_rows: print(r)
    print(summary["intro_africa_100_vs_300"])

    plot(b1, pairs, sens)
    print(json.dumps({"replication": replication, "comparison": comparison}, ensure_ascii=False, indent=2))
    for p in pairs: print(p)
    for s in sens: print(s)


def plot(base_rows, pairs, sens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ep = [r["episode"] for r in base_rows]
    rew = np.array([r["reward"] for r in base_rows])
    ma = np.convolve(rew, np.ones(10) / 10, mode="valid")
    ax[0].plot(ep, rew, color="#9bb7d4", lw=0.8, label="episode reward")
    ax[0].plot(ep[9:], ma, color="#1f4e79", lw=1.8, label="10-episode moving avg")
    ax[0].axvspan(*WINDOW, color="#f4b183", alpha=0.3, label=f"eval window {WINDOW[0]}-{WINDOW[1]}")
    ax[0].set(title="(a) Baseline: AFRICA opt E.15 S.40 M.25 I.20, seed 42", xlabel="episode", ylabel="total reward")
    ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=.3)

    x = np.arange(len(pairs))
    ax[1].bar(x - 0.18, [p["A_div"] for p in pairs], 0.36, label=f"A: identity raw {LEVELS['A']:.2f}", color="#7f7f7f")
    ax[1].bar(x + 0.18, [p["B_div"] for p in pairs], 0.36, label=f"B: identity raw {LEVELS['B']:.2f}", color="#2e75b6")
    ax[1].set_xticks(x, [str(p["seed"]) for p in pairs])
    ax[1].set(title=f"(b) Paired seeds: mean diversity, ep {WINDOW[0]}-{WINDOW[1]}", xlabel="seed", ylabel="diversity (0=one-sided, 1=50/50)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")

    lv = [s["identity_raw"] for s in sens]
    dm = np.array([s["diversity_mean"] for s in sens]); dsd = np.array([s["diversity_sd"] for s in sens])
    ax[2].errorbar(lv, dm, yerr=dsd, marker="o", color="#1f4e79", capsize=3, label="diversity (mean ± SD, 5 seeds)")
    for s in sens:
        ax[2].annotate(s["greedy_choice"], (s["identity_raw"], s["diversity_mean"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax[2].axvspan(0.20, 0.30, color="#c5e0b4", alpha=.5, label="AFRICA band I .20-.30")
    ax[2].set(title="(c) Sensitivity: identity raw level (label = best option S1 S2 S3, = tie)", xlabel="identity raw value (others fixed)", ylabel="diversity")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_w1_result.png", dpi=150)


if __name__ == "__main__":
    main()
