"""Global AI Ethics Simulator, Streamlit 진입점."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
FRAMES = ("emotion", "social", "moral", "identity")

st.set_page_config(page_title="Global AI Ethics Simulator", layout="wide")
st.title("Global AI Ethics Simulator")
st.caption("문화권 기반 윤리 강화학습 · 팀 AFRICA")

st.header("문화권 프리셋")
presets = {}
for path in sorted((ROOT / "presets").glob("*.yaml")):
    if path.name.startswith("_"):
        continue
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    presets[doc.get("meta", {}).get("culture_code", path.stem)] = doc

if presets:
    rows = {code: {k: d.get("weights", {}).get(k) for k in FRAMES} for code, d in presets.items()}
    df = pd.DataFrame(rows).T
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df)
    for code, d in presets.items():
        with st.expander(f"{code} 근거와 한계"):
            st.markdown("**Rationale**")
            st.write(d.get("rationale", ""))
            st.markdown("**Sources**")
            for s in d.get("sources", []):
                st.markdown(f"- [{s.get('title') or s.get('url')}]({s.get('url')})")
            st.markdown("**Limits**")
            for item in d.get("limits", []):
                st.markdown(f"- {item}")
else:
    st.info("아직 프리셋이 없습니다. W1에서 presets/AFRICA.yaml 을 추가하세요.")

st.header("시나리오")
scen_path = ROOT / "data" / "scenarios.json"
scenarios = json.loads(scen_path.read_text(encoding="utf-8")).get("scenarios", []) if scen_path.exists() else []
if scenarios:
    for s in scenarios:
        with st.expander(f"{s['id']} · {s['title']}"):
            st.write(s["context"])
            st.dataframe(pd.DataFrame({k: v["reward"] for k, v in s["options"].items()}).T)
else:
    st.info("아직 시나리오가 없습니다. W2에서 data/scenarios.json 을 채우세요.")

st.header("강화학습 결과 (W3)")
summary_path = ROOT / "results" / "summary.json"
if not summary_path.exists():
    st.info("아직 결과가 없습니다. `python -m rl.agent` 를 실행하세요.")
    st.stop()

from rl.agent import ACTIONS, SEEDS, run_one  # noqa: E402

S = json.loads(summary_path.read_text(encoding="utf-8"))
cfg = S["config"]
st.caption(
    f"ε-greedy Q-learning (bandit) · {len(S['weights'])} 문화 × {len(cfg['seeds'])} seeds × {cfg['episodes']} episodes · "
    f"α={cfg['alpha']} · ε {cfg['eps_start']}→{cfg['eps_min']} (decay {cfg['eps_decay']}, 500회 후 ε≈{cfg['eps_final']}) · "
    f"지표는 마지막 {cfg['tail']} episode 기준, 평균±SD (n={len(cfg['seeds'])} seeds)"
)
nonteam = [c for c, src in S["weights_source"].items() if "reference" in src]
if nonteam:
    st.warning("다음 문화는 팀 프리셋이 아니라 강의자료 예시 가중치로 실행했습니다: " + ", ".join(nonteam))

def pm(m, nd=3):
    return f"{m['mean']:.{nd}f} ± {m['sd']:.{nd}f}"

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Reward", "Entropy", "Diversity", "Pearson r", "직접 실험"])

with tab1:
    pc = S["per_culture"]
    df = pd.DataFrame({c: {"Reward (평균±SD)": pm(m["reward"]), "reward_mean": m["reward"]["mean"],
                           "reward_sd": m["reward"]["sd"], "A 선택 시나리오 비율": m["share_A_choices"]}
                       for c, m in pc.items()}).T
    st.dataframe(df[["Reward (평균±SD)", "A 선택 시나리오 비율"]], use_container_width=True)
    st.bar_chart(df["reward_mean"].astype(float))
    st.caption("mean(R[400:500]). 프리셋마다 보상 스케일이 달라 문화 간 크기 비교는 조심해야 합니다.")
    sid = st.selectbox("시나리오별 보기", S["scenarios"], key="rew_s")
    st.dataframe(pd.DataFrame({c: {"선택": S["cells"][c][sid]["choice"], "Reward": pm(S["cells"][c][sid]["reward"]),
                                   "p(A)": pm(S["cells"][c][sid]["p_A"], 2)} for c in S["cells"]}).T,
                 use_container_width=True)

with tab2:
    ent = pd.DataFrame({c: {sid: S["cells"][c][sid]["entropy"]["mean"] for sid in S["scenarios"]} for c in S["cells"]}).T
    st.dataframe(ent.round(3), use_container_width=True)
    st.caption("H = −Σ p·log₂p. 1이면 50:50, 0이면 한쪽으로 확정. 남은 탐험(ε≈0.08) 때문에 0까지 내려가지 않습니다.")
    st.subheader("선택 분포 (p(A), 마지막 100 episode)")
    pa = pd.DataFrame({c: {sid: S["cells"][c][sid]["p_A"]["mean"] for sid in S["scenarios"]} for c in S["cells"]}).T
    st.bar_chart(pa)

with tab3:
    st.dataframe(pd.DataFrame({sid: {"D": d["D"], **d["choices"]} for sid, d in S["diversity"].items()}).T,
                 use_container_width=True)
    st.caption("D = #distinct_choices / 7. 옵션이 A/B 두 개라 D는 1/7(=0.143, 모두 같은 답) 또는 2/7(=0.286)만 나옵니다.")

with tab4:
    pooled = pd.DataFrame(S["pearson"]["pooled"]).T
    st.dataframe(pooled, use_container_width=True)
    rows = []
    for sid, d in S["pearson"]["per_scenario"].items():
        for k, v in d.items():
            rows.append({"scenario": sid, "frame": k, "frame 유리 옵션": v["favored"], "r": v["r"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("r = corr(w_frame, 그 프레임 보상이 더 큰 옵션을 고를 확률), 7문화. n이 작아 검정력이 낮고, 상관은 인과가 아닙니다.")

with tab5:
    st.write("가중치를 바꿔 5 seeds로 바로 다시 학습해 봅니다. 결과 파일은 바뀌지 않습니다.")
    base = st.selectbox("시작 프리셋", list(S["weights"]), index=len(S["weights"]) - 1)
    cols = st.columns(4)
    w = {k: cols[i].slider(k, 0.0, 0.5, float(S["weights"][base][k]), 0.01) for i, k in enumerate(FRAMES)}
    total = sum(w.values())
    if abs(total - 1.0) > 0.005:
        st.error(f"가중치 합 = {total:.2f} (1.00 이어야 합니다)")
    else:
        res = [r for sd in SEEDS for r in run_one("custom", sd, w, scenarios)]
        out = {}
        for sid in S["scenarios"]:
            rs = [r for r in res if r["scenario"] == sid]
            g = [r["greedy"] for r in rs]
            out[sid] = {"선택": max(ACTIONS, key=g.count),
                        "p(A)": f"{pd.Series([r['p_A'] for r in rs]).mean():.2f}",
                        "Reward": f"{pd.Series([r['reward_mean'] for r in rs]).mean():+.3f}",
                        "원래 선택": S["cells"][base][sid]["choice"]}
        st.dataframe(pd.DataFrame(out).T, use_container_width=True)

st.divider()
st.caption("이 결과는 우리 팀이 정한 가중치와 시나리오 위에서 학습기가 어떻게 행동하는지 보여주는 실험이며, 어떤 문화의 판단을 대표하지 않습니다.")
