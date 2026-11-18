"""logs/w3 의 seed 별 CSV 가 코드 재실행과 같은지 확인 (워크북 방식 재현성)."""
import csv
from pathlib import Path

import pytest

from rl.agent import load_weights
from rl.scenario_io import load_normalized
from rl.workbook_protocol import OUT, run


@pytest.mark.parametrize("culture,seed", [("AFRICA", 42), ("KOREA", 2026), ("USA", 31337)])
def test_logged_csv_matches_rerun(culture, seed):
    path = OUT / f"{culture}_seed{seed}_ep500.csv"
    logged = list(csv.DictReader(path.open(encoding="utf-8")))
    weights, _ = load_weights(culture)
    hist, _ = run(weights, load_normalized(), seed)
    assert len(logged) == len(hist) == 500
    for a, b in zip(logged, hist):
        assert abs(float(a["reward"]) - b["reward"]) < 1e-5
        assert all(a[k] == b[k] for k in b if k.startswith("act_"))
