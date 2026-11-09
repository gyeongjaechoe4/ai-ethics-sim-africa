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

st.header("강화학습 결과")
st.info("W3에서 rl/agent.py 학습 결과(Reward·Entropy·Diversity·Pearson r)를 여기에 표시합니다.")
