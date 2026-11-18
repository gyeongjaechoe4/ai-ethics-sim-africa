# Global AI Ethics Simulator · Team AFRICA

![Python](https://img.shields.io/badge/python-3.11-blue)
![CI](https://github.com/gyeongjaechoe4/ai-ethics-sim-africa/actions/workflows/validate.yml/badge.svg)

인공지능과 경영(2026 Fall, 강송희 교수) "문화권 기반 윤리 강화학습 3주 스프린트" 실습 저장소.
문화권 프리셋(E·S·M·I 4프레임 가중치) → 딜레마 시나리오 → ε-greedy Q-learning → Streamlit 시연.

**Streamlit Cloud URL**: _(배포 후 기입)_ · Main file `app_reinforce.py` · Python 3.11

## 🌏 팀 정보

| 항목 | 내용 |
|---|---|
| 팀명 | T? |
| 배정 문화권 | AFRICA |
| Streamlit Cloud URL | (배포 후 기입) |

## 👥 팀원 (역할은 주차마다 로테이션)

| 이름 | 학번 | 전공 | 역할 (W1 / W2 / W3) | GitHub |
|---|---|---|---|---|
| | | | PO · RL Engineer · Narrative · Scribe | |
| | | | | |
| | | | | |
| | | | | |

## 📅 진행 상태

- [ ] **Week 1** — `docs/week1/culture_report.md` + `presets/AFRICA.yaml` (팀이 Non-AI 로 작성)
- [x] **Week 2** — `data/scenarios.json` 4건 (PR #1) · [ ] 타팀 Value-Check 2건 · [ ] 받은 리뷰 반영
- [x] **Week 3** — 시드 스윕 2종(슬라이드·워크북), `audit.md`, `docs/week3/final_report.md` 수치 · [ ] 가설 검증·재설계·Reflection (팀)

## 기준 문서

교수님 자료가 두 벌이라 둘 다 맞춘다.
- 슬라이드 기준: `presets/AFRICA.yaml`, `data/scenarios.json`(id `AF-XX-01`), `audit.md`, seeds 0~4
- student_package(워크북·채점표·템플릿) 기준: `docs/week1~3/*.md`, `logs/ai_usage_log.md`, `data/scenarios.template.json`(sid `S1_xxx`), `logs/w3/`, seeds 42·123·2026·7777·31337

## 구조
```
app_reinforce.py          Streamlit 진입점 (교수님 앱 + 팀 시나리오·프리셋 선택)
app.py                    W3 결과 뷰어 (슬라이드 방식 results/)
presets/_TEMPLATE.yaml    W1 프리셋 빈 양식 → presets/AFRICA.yaml (팀이 LLM 없이 작성)
data/scenarios.json       W2 시나리오 (슬라이드 형식, 원본)
data/scenarios.template.json  같은 시나리오의 템플릿 형식 (python -m rl.scenario_io 로 생성)
rl/agent.py               슬라이드 방식 학습기 → results/
rl/workbook_protocol.py   워크북 방식 시드 스윕 → logs/w3/ (seed별 CSV, summary.json, 그림 3장)
rl/scenario_io.py         두 시나리오 형식 로더·변환기
rl/reference_presets.yaml 강의 슬라이드 예시 가중치 (presets/<CODE>.yaml 없을 때 대체값)
docs/week1~3/             culture_report · scenario_review · final_report · draft_no_ai
docs/scrum/, docs/reflection/  스크럼 회의록, 개인 성찰 (템플릿)
logs/ai_usage_log.md      AI 사용 로그 (Non-AI vs AI diff)
audit.md                  W3 감사 보고서 (슬라이드 기준)
schemas/, scripts/        스키마와 검증기 (validate.py = 슬라이드, validate_scenarios.py = 템플릿)
tests/                    재현성·형식 테스트 (pytest, CI)
```

## 로컬 실행 (Python 3.11)
```bash
git clone https://github.com/gyeongjaechoe4/ai-ethics-sim-africa.git && cd ai-ethics-sim-africa
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app_reinforce.py   # 교수님 앱 (http://localhost:8501)
python scripts/validate.py       # 슬라이드 스키마 검증
python -m rl.scenario_io --team T?   # 템플릿 형식 내보내기 (팀 번호로)
python scripts/validate_scenarios.py --file data/scenarios.template.json
python -m rl.agent               # 슬라이드 방식 7문화 × 5seed → results/
python -m rl.workbook_protocol   # 워크북 방식 7문화 × 5seed → logs/w3/
pytest -q                        # CI 와 같은 테스트
```

## 협업 규칙
- main 직접 push 금지, PR 최소 1인 승인 후 머지 (dev → main)
- 브랜치 `feat/`, `fix/`, `docs/`, `exp/` · 커밋 `type(scope): [Wn] 요약` (Conventional Commits + 주차 태그)
- 매 주차 Non-AI 초안(`docs/weekN/draft_no_ai.md`) → AI 사용 시 `logs/ai_usage_log.md` 에 기록
- 스크럼 회의록 주 1회 `docs/scrum/`
- 제출 24시간 전 다른 PC 에서 새 clone → 설치 → 500 episode 재현 리허설

## 배포
Streamlit Cloud: New app → 이 repo · branch `main` · Main file `app_reinforce.py` · Advanced settings에서 Python 3.11.
API 키 등 민감 정보는 코드가 아니라 Streamlit Secrets에 둔다 (`.streamlit/secrets.toml` 은 .gitignore).

## 출처
- 강송희, "Global AI Ethics Simulator — 문화권 기반 윤리 강화학습 3주 스프린트", 인공지능과 경영 강의자료·학생 워크북 v1.0, 2026. `app_reinforce.py` 원본 저작자 강송희 교수 (CC BY-NC 4.0).
- Hofstede Insights, Country Comparison: https://www.hofstede-insights.com/country-comparison-tool
- World Values Survey Wave 7: https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp
- 시나리오별 근거 URL은 `data/scenarios.json` 의 `sources`.
