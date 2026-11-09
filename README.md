# Global AI Ethics Simulator · Team AFRICA

인공지능과 경영(2026 Fall, 강송희 교수) "문화권 기반 윤리 강화학습 3주 스프린트" 실습 저장소.
문화권 프리셋(E·S·M·I 4프레임 가중치) → 딜레마 시나리오 → ε-greedy Q-learning → Streamlit 시연.

## 구조
```
presets/_TEMPLATE.yaml   W1 프리셋 빈 양식 (복사해서 AFRICA.yaml 작성, CI 제외)
presets/AFRICA.yaml      W1 산출물 (팀이 LLM 없이 직접 작성)
data/scenarios.json      W2 산출물 (시나리오 ≥3건)
rl/agent.py              W3 학습기
app.py                   Streamlit 진입점
audit.md                 W3 감사 보고서
schemas/                 프리셋·시나리오 JSON Schema
scripts/validate.py      CI 검증 스크립트 (스키마 + Σw=1.00±0.005)
.github/                 Actions CI, PR 템플릿, Value-Check 이슈 템플릿
```

## 로컬 실행 (Python 3.11)
```bash
git clone <repo-url> && cd <repo>
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/validate.py       # 스키마 검증
streamlit run app.py             # http://localhost:8501
```

## 협업 규칙 (7 Steps)
1. PLAN: Issue 생성 → 2. CODE: `git checkout -b feat/africa-preset` (dev에서 분기)
3. COMMIT: Conventional Commits (`feat(presets): add AFRICA draft`, `fix(rl): clamp reward to [-1,1]`, `docs(readme): ...`)
4. PUSH: `git push origin feat/...` → 5. PR: dev 대상, 템플릿·근거 링크 필수
6. REVIEW: 리뷰 승인 1건 이상 + CI 통과 → 7. MERGE: dev → main (main 병합 시 Streamlit 자동 재배포)

## 배포
Streamlit Cloud: New app → 이 repo · branch `main` · `app.py`, Advanced settings에서 Python 3.11.
API 키 등 민감 정보는 코드가 아니라 Streamlit Secrets에 둔다.

배포 URL: _(배포 후 기입)_

## 출처
- 강송희, "Global AI Ethics Simulator — 문화권 기반 윤리 강화학습 3주 스프린트", 인공지능과 경영 강의자료, 2026.
- Hofstede Insights, Country Comparison: https://www.hofstede-insights.com/country-comparison-tool
- World Values Survey Wave 7: https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp
