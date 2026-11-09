"""스키마 검증 CI 스크립트: presets/*.yaml 과 data/scenarios.json 을 검사한다.

사용법: python scripts/validate.py
"""
import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent
SUM_TOL = 0.005
FRAMES = ("emotion", "social", "moral", "identity")


def load_schema(name):
    return Draft202012Validator(json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8")))


def check_presets():
    errors = []
    validator = load_schema("preset.schema.json")
    for path in sorted((ROOT / "presets").glob("*.yaml")):
        if path.name.startswith("_"):
            continue  # 템플릿은 건너뜀
        rel = path.relative_to(ROOT)
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:
            errors.append(f"{rel}: YAML 파싱 실패: {e}")
            continue
        for err in validator.iter_errors(doc):
            loc = "/".join(str(p) for p in err.absolute_path) or "(root)"
            errors.append(f"{rel}: {loc}: {err.message}")
        weights = (doc or {}).get("weights") or {}
        if all(isinstance(weights.get(k), (int, float)) for k in FRAMES):
            total = sum(weights[k] for k in FRAMES)
            if abs(total - 1.0) > SUM_TOL:
                errors.append(f"{rel}: weights 합 = {total:.3f} (1.00 ± {SUM_TOL} 이어야 함)")
            if sum(weights[k] >= 0.10 for k in FRAMES) < 3:
                print(f"[경고] {rel}: 0.10 이상인 축이 3개 미만 (권장 사항)")
        print(f"[검사] {rel}")
    return errors


def check_scenarios():
    path = ROOT / "data" / "scenarios.json"
    if not path.exists():
        return []
    rel = path.relative_to(ROOT)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"{rel}: JSON 파싱 실패: {e}"]
    errors = []
    for err in load_schema("scenarios.schema.json").iter_errors(doc):
        loc = "/".join(str(p) for p in err.absolute_path) or "(root)"
        errors.append(f"{rel}: {loc}: {err.message}")
    ids = [s.get("id") for s in doc.get("scenarios", []) if isinstance(s, dict)]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        errors.append(f"{rel}: 중복 id {sorted(dupes)}")
    if len(ids) < 3:
        print(f"[경고] {rel}: 시나리오 {len(ids)}건 (W2 제출 시 3건 이상 필요)")
    print(f"[검사] {rel}")
    return errors


def main():
    errors = check_presets() + check_scenarios()
    if errors:
        print("\n검증 실패:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("\n모든 검증 통과")


if __name__ == "__main__":
    main()
