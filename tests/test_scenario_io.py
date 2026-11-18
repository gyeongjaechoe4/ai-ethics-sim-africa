"""템플릿 형식(student_package) 내보내기가 최신이고 템플릿 검증기를 통과하는지 확인."""
import json
import subprocess
import sys
from pathlib import Path

from rl.scenario_io import load_normalized, to_template

ROOT = Path(__file__).resolve().parent.parent
TPL = ROOT / "data" / "scenarios.template.json"


def test_template_export_is_up_to_date():
    committed = json.loads(TPL.read_text(encoding="utf-8"))
    src = json.loads((ROOT / "data" / "scenarios.json").read_text(encoding="utf-8"))
    fresh = to_template(src, committed["culture"], committed["team"], committed.get("weights"))
    assert committed == fresh, "data/scenarios.json 이 바뀌었으면 `python -m rl.scenario_io --team <팀번호>` 를 다시 실행"


def test_template_export_passes_template_validator():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_scenarios.py"), "--file", str(TPL),
                        "--schema", str(ROOT / "schemas" / "scenario.template.schema.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_both_formats_normalize_to_same_rewards():
    a = load_normalized(ROOT / "data" / "scenarios.json")
    b = load_normalized(TPL)
    assert [s["rewards"] for s in a] == [s["rewards"] for s in b]
    assert [s["options"] for s in a] == [s["options"] for s in b]
