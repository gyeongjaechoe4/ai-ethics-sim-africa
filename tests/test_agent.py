"""W3 재현성 테스트: CI(GitHub Actions)에서 seed 3개를 다시 돌려 results/runs.csv 와 비교한다."""
import csv
import json

import pytest

from rl.agent import CONFIG, RESULTS, SEEDS, load_scenarios, load_weights, reward, run_one, train

SUMMARY = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
with (RESULTS / "runs.csv").open(encoding="utf-8") as f:
    RUNS = list(csv.DictReader(f))


def test_reward_is_weighted_sum():
    w = {"emotion": 0.25, "social": 0.35, "moral": 0.25, "identity": 0.15}
    r = {"emotion": -0.4, "social": 0.3, "moral": 0.7, "identity": 0.5}
    assert reward(w, r) == pytest.approx(0.255)  # 강의자료 KOREA·A 예시


def test_same_seed_same_result():
    w, _ = load_weights("AFRICA")
    a = train(w, load_scenarios(), seed=3)
    b = train(w, load_scenarios(), seed=3)
    assert (a["choices"] == b["choices"]).all()
    assert (a["Q"] == b["Q"]).all()


def test_epsilon_schedule():
    w, _ = load_weights("USA")
    eps = train(w, load_scenarios(), seed=0)["eps"]
    assert eps[0] == 1.0 and eps.min() >= CONFIG["eps_min"]
    assert len(eps) == CONFIG["episodes"]


def test_full_sweep_recorded():
    assert len(SUMMARY["weights"]) == 7 and list(SUMMARY["config"]["seeds"]) == list(SEEDS)
    assert len(RUNS) == 7 * len(SEEDS) * len(SUMMARY["scenarios"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_report_matches_rerun(seed):
    """results/ 에 기록된 가중치로 다시 돌리면 같은 숫자가 나와야 한다 (보고서와 다르면 실패)."""
    scenarios = load_scenarios()
    assert [s["id"] for s in scenarios] == SUMMARY["scenarios"], "시나리오가 바뀌었으면 python -m rl.agent 재실행"
    for culture, w in SUMMARY["weights"].items():
        fresh = run_one(culture, seed, w, scenarios)
        saved = [r for r in RUNS if r["culture"] == culture and int(r["seed"]) == seed]
        for f, s in zip(fresh, saved):
            assert f["greedy"] == s["greedy"]
            assert f["reward_mean"] == pytest.approx(float(s["reward_mean"]), abs=1e-6)
            assert f["p_A"] == pytest.approx(float(s["p_A"]), abs=1e-6)
