"""scenarios.json 두 형식을 함께 다루는 도우미.

형식 1 (강의 슬라이드, 우리 data/scenarios.json):
    {"version", "scenarios": [{"id", "title", "context", "options": {"A": {"label", "reward"}, ...}, "sources", ...}]}
형식 2 (student_package 템플릿, schemas/scenario.schema.json):
    {"culture", "team", "weights"?, "scenarios": [{"sid", "title", "setup", "options": {"A": str}, "rewards": {"A": {...}}}]}

load_normalized() 는 어느 형식이든 형식 2 모양의 시나리오 리스트로 돌려준다.
to_template() 은 형식 1 문서를 형식 2 문서로 바꾼다 (추가 키는 템플릿 스키마가 금지하므로 뺀다).

실행: python -m rl.scenario_io --team T0   -> data/scenarios.template.json 생성
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ("emotion", "social", "moral", "identity")
SRC = ROOT / "data" / "scenarios.json"
OUT = ROOT / "data" / "scenarios.template.json"


def _sid(index: int, legacy_id: str) -> str:
    """AF-DP-01 -> S1_af_dp_01 (템플릿 정규식 ^S[0-9]+(_[a-z0-9]+)?$ 는 밑줄 1개만 허용하므로 접미사는 붙여 씀)"""
    suffix = re.sub(r"[^a-z0-9]", "", legacy_id.lower())
    return f"S{index}_{suffix}" if suffix else f"S{index}"


def normalize(doc: dict) -> list[dict]:
    out = []
    for i, s in enumerate(doc.get("scenarios", []), start=1):
        if "sid" in s:  # 이미 형식 2
            out.append({"sid": s["sid"], "title": s["title"], "setup": s["setup"],
                        "options": {a: s["options"][a] for a in ("A", "B")},
                        "rewards": {a: {k: float(s["rewards"][a][k]) for k in FRAMES} for a in ("A", "B")},
                        "source_id": s["sid"]})
        else:  # 형식 1
            out.append({"sid": _sid(i, s["id"]), "title": s["title"], "setup": s["context"],
                        "options": {a: s["options"][a]["label"] for a in ("A", "B")},
                        "rewards": {a: {k: float(s["options"][a]["reward"][k]) for k in FRAMES} for a in ("A", "B")},
                        "source_id": s["id"]})
    return out


def load_normalized(path: Path = SRC) -> list[dict]:
    return normalize(json.loads(Path(path).read_text(encoding="utf-8")))


def to_template(doc: dict, culture: str = "AFRICA", team: str = "T0", weights: dict | None = None) -> dict:
    out = {"culture": culture, "team": team}
    if weights:
        out["weights"] = {k: float(weights[k]) for k in FRAMES}
    out["scenarios"] = [{k: s[k] for k in ("sid", "title", "setup", "options", "rewards")} for s in normalize(doc)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="data/scenarios.json -> 템플릿 형식 내보내기")
    ap.add_argument("--team", default="T0", help="팀 번호 (예: T3). 배정받은 번호로 바꿀 것")
    ap.add_argument("--culture", default="AFRICA")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    doc = json.loads(SRC.read_text(encoding="utf-8"))
    weights = None
    preset = ROOT / "presets" / f"{args.culture}.yaml"
    if preset.exists():  # 팀 프리셋이 있으면 weights 도 함께 기록
        import yaml
        weights = yaml.safe_load(preset.read_text(encoding="utf-8"))["weights"]
    tpl = to_template(doc, args.culture, args.team, weights)
    Path(args.out).write_text(json.dumps(tpl, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.out}: culture={args.culture}, team={args.team}, 시나리오 {len(tpl['scenarios'])}개"
          + (", weights 포함" if weights else ", weights 없음 (presets/%s.yaml 없음)" % args.culture))


if __name__ == "__main__":
    main()
