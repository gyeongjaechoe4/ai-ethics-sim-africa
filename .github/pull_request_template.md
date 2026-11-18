## 📌 PR 개요

<!-- 이 PR이 해결하는 문제/추가하는 기능을 2~3줄로 요약 -->

## 🗓️ 스프린트 주차

- [ ] Week 1 — 문화권 기반 초기 전략 설계
- [ ] Week 2 — 시나리오 설계 및 보상 구조화
- [ ] Week 3 — 시뮬레이션 실행 · 감사 · 재설계

## 🔗 근거 링크 · 관련 Issue

<!-- 가중치·시나리오 근거 URL (Hofstede, WVS, 뉴스, 판례, 논문), closes #N -->
-

## ✅ 과정증거 체크리스트 (P-A-S-A-R)

- [ ] **Non-AI 초안** 이 `docs/weekN/draft_no_ai.md` 에 커밋되어 있다 (W1 프리셋은 `draft_mode: "non-ai"`)
- [ ] **AI 사용 로그** `logs/ai_usage_log.md` 에 프롬프트·응답·수정 흔적 3-entry 이상
- [ ] **검증 통과**: `python scripts/validate.py`, `python scripts/validate_scenarios.py --file data/scenarios.template.json`, `pytest -q` (CI 그린)
- [ ] **리뷰어** 최소 1명 이상 배정
- [ ] **재현성** seed 값을 파일명/커밋 메시지에 명시
- [ ] 브랜치 `feat/ fix/ docs/ exp/`, 커밋 `type(scope): [Wn] 요약`

## 🔬 실험 정보 (Week3 PR만 해당)

| 항목 | 값 |
|---|---|
| seed(s) | 42, 123, 2026, 7777, 31337 (워크북) / 0~4 (슬라이드) |
| episodes | 500 |
| culture(s) | |

## 📎 스크린샷 / 그래프

## 💬 리뷰어 요청 포인트
