#!/usr/bin/env python3
"""
validate_scenarios.py
Global AI Ethics Simulator — Week2 산출물 검증기

검증 항목:
  1. JSON Schema (draft-07) 구조 검증
  2. 시나리오 최소 3개
  3. culture 화이트리스트 검증
  4. sid 중복 검사
  5. 옵션 A/B 존재 및 텍스트 길이
  6. 보상값 [-1.0, 1.0] 범위 재검증

사용:
  python scripts/validate_scenarios.py --file data/scenarios.json
  python scripts/validate_scenarios.py --file data/scenarios.json --schema schemas/scenario.schema.json

exit code:
  0 = 통과
  1 = 검증 실패
  2 = 파일 I/O 오류
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except ImportError:
    print("[ERROR] jsonschema 미설치. `pip install jsonschema` 실행 후 재시도.", file=sys.stderr)
    sys.exit(2)


ALLOWED_CULTURES = {"USA", "CHINA", "EUROPE", "KOREA", "LATIN_AM", "MIDDLE_E", "AFRICA"}
FRAMEWORKS = ("emotion", "social", "moral", "identity")


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] 파일이 존재하지 않습니다: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 실패: {e}", file=sys.stderr)
        sys.exit(2)


def check_schema(data: dict, schema: dict) -> list[str]:
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    return [f"schema: {'/'.join(map(str, e.path)) or '<root>'} — {e.message}" for e in errors]


def check_domain_rules(data: dict) -> list[str]:
    errs: list[str] = []

    # 규칙 1: culture 화이트리스트
    culture = data.get("culture")
    if culture not in ALLOWED_CULTURES:
        errs.append(f"culture '{culture}' 는 허용 목록에 없음: {sorted(ALLOWED_CULTURES)}")

    scenarios = data.get("scenarios", [])

    # 규칙 2: 시나리오 최소 3개
    if len(scenarios) < 3:
        errs.append(f"시나리오는 최소 3개가 필요합니다 (현재 {len(scenarios)}개)")

    # 규칙 3: sid 중복 검사
    sids = [s.get("sid") for s in scenarios]
    dup = {sid for sid in sids if sids.count(sid) > 1}
    if dup:
        errs.append(f"중복된 sid: {sorted(dup)}")

    # 규칙 4·5: 옵션 A/B 및 보상 범위
    for i, sc in enumerate(scenarios):
        sid = sc.get("sid", f"index{i}")

        opts = sc.get("options", {})
        for key in ("A", "B"):
            if key not in opts:
                errs.append(f"[{sid}] options.{key} 누락")
            elif not isinstance(opts[key], str) or len(opts[key]) < 2:
                errs.append(f"[{sid}] options.{key} 텍스트가 너무 짧음")

        rewards = sc.get("rewards", {})
        for opt in ("A", "B"):
            if opt not in rewards:
                errs.append(f"[{sid}] rewards.{opt} 누락")
                continue
            for fw in FRAMEWORKS:
                v = rewards[opt].get(fw)
                if not isinstance(v, (int, float)):
                    errs.append(f"[{sid}] rewards.{opt}.{fw} 는 숫자여야 함")
                elif not (-1.0 <= float(v) <= 1.0):
                    errs.append(f"[{sid}] rewards.{opt}.{fw} = {v} 는 [-1.0, 1.0] 범위 밖")

    # 규칙 6: weights 합 = 1.0 (오차 1e-3)
    w = data.get("weights")
    if w:
        total = sum(w.get(fw, 0.0) for fw in FRAMEWORKS)
        if abs(total - 1.0) > 1e-3:
            errs.append(f"weights 합 = {total:.4f}, 반드시 1.0 이어야 합니다.")

    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description="Ethics Agent Game 시나리오 검증기")
    parser.add_argument("--file",   default="data/scenarios.json",         help="검증할 JSON 파일")
    parser.add_argument("--schema", default="schemas/scenario.template.schema.json", help="JSON Schema 파일")
    args = parser.parse_args()

    data   = load_json(Path(args.file))
    schema = load_json(Path(args.schema))

    all_errors = check_schema(data, schema) + check_domain_rules(data)

    if all_errors:
        print(f"❌ 검증 실패 ({len(all_errors)}건)")
        for i, msg in enumerate(all_errors, 1):
            print(f"  {i}. {msg}")
        return 1

    n = len(data.get("scenarios", []))
    print(f"✅ 검증 통과 — culture={data.get('culture')}, team={data.get('team')}, 시나리오 {n}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
