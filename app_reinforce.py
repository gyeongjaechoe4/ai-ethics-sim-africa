# app.py — TU Korea AI Management: Ethical AI Simulation (with Strategy Entropy)
# Author: Prof. Songhee Kang
# Repository-optimized version: reproducibility, typing, English UI, stability improvements

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import random

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, entropy as scipy_entropy


Action = Literal["A", "B"]
FRAMEWORKS: Tuple[str, ...] = ("emotion", "social", "moral", "identity")


# =========================
# 1. Page configuration
# =========================
st.set_page_config(
    page_title="(TU Korea) Ethical AI Agent Reinforcement Learning Simulation",
    page_icon="🎓",
    layout="wide",
)


# =========================
# 2. Environment model
# =========================
@dataclass(frozen=True)
class Scenario:
    """A single ethical dilemma scenario."""
    sid: str
    title: str
    setup: str
    options: Dict[Action, str]
    rewards: Dict[Action, Dict[str, float]]  # framework -> reward in [-1.0, 1.0]


@st.cache_data(show_spinner=False)
def get_default_scenarios() -> List[Scenario]:
    """Default scenarios used in the paper/demo."""
    return [
        Scenario(
            sid="S1",
            title="Stage 1: Classic Trolley Problem",
            setup="5 people on the track vs 1 person. Do you pull the lever?",
            options={"A": "Sacrifice 1 person (Intervene)", "B": "Do nothing (Status quo)"},
            rewards={
                "A": {"emotion": 1.0, "social": -0.5, "moral": -1.0, "identity": 0.5},
                "B": {"emotion": -1.0, "social": 0.5, "moral": 1.0, "identity": -0.5},
            },
        ),
        Scenario(
            sid="S2",
            title="Stage 2: Contextual Factors",
            setup="5 trespassers vs 1 child of the facility manager.",
            options={"A": "Save 5 people (child dies)", "B": "Follow the rule (let 5 die)"},
            rewards={
                "A": {"emotion": 0.6, "social": -0.8, "moral": -0.7, "identity": 0.3},
                "B": {"emotion": -0.5, "social": 0.9, "moral": 0.6, "identity": 0.4},
            },
        ),
        Scenario(
            sid="S3",
            title="Stage 3: Medical Triage",
            setup="Many injured civilians vs 1 highly skilled doctor.",
            options={"A": "Prioritize the doctor (Utilitarian)", "B": "Treat equally (Egalitarian)"},
            rewards={
                "A": {"emotion": 0.7, "social": -0.4, "moral": -0.6, "identity": 0.8},
                "B": {"emotion": -0.3, "social": 0.7, "moral": 0.9, "identity": 0.5},
            },
        ),
    ]


@st.cache_data(show_spinner=False)
def get_culture_presets() -> Dict[str, Dict[str, float]]:
    """Cultural weight presets (must sum to 1 after normalization)."""
    return {
        "USA":      {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
        "CHINA":    {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
        "EUROPE":   {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
        "KOREA":    {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
        "LATIN_AM": {"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
        "MIDDLE_E": {"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
        "AFRICA":   {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
    }


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values())) or 1.0
    return {k: float(v) / total for k, v in weights.items()}


# =========================
# 3. Simple E-Greedy agent
# =========================
class SimpleEGreedyAgent:
    """
    A minimal E-Greedy reinforcement learning agent.

    - Maintains a Q-table per scenario (A/B).
    - Selects actions via epsilon-greedy exploration.
    - Updates Q-values using incremental mean style update:
        Q <- Q + lr * (reward - Q)
    """

    def __init__(
        self,
        name: str,
        weights: Dict[str, float],
        scenarios: List[Scenario],
        learning_rate: float = 0.1,
        epsilon: float = 0.5,
    ) -> None:
        self.name = name
        self.weights = weights
        self.scenarios = scenarios
        self.lr = float(learning_rate)
        self.epsilon = float(epsilon)

        self.q_table: Dict[str, Dict[Action, float]] = {
            s.sid: {"A": 0.0, "B": 0.0} for s in scenarios
        }

    def get_action(self, sid: str) -> Action:
        # 1) Exploration
        if random.random() < self.epsilon:
            return random.choice(["A", "B"])

        # 2) Exploitation
        qs = self.q_table[sid]
        if qs["A"] > qs["B"]:
            return "A"
        if qs["B"] > qs["A"]:
            return "B"
        return random.choice(["A", "B"])

    def calculate_reward(self, sid: str, action: Action) -> float:
        # Reward = dot(scenario_reward_vector, culture_weight_vector) * 10
        scn = next(s for s in self.scenarios if s.sid == sid)
        r_vec = scn.rewards[action]
        reward = sum(float(r_vec.get(k, 0.0)) * float(self.weights.get(k, 0.0)) for k in FRAMEWORKS) * 10.0
        return float(reward)

    def update(self, sid: str, action: Action, reward: float) -> None:
        old_val = float(self.q_table[sid][action])
        self.q_table[sid][action] = old_val + self.lr * (float(reward) - old_val)

    def decay_epsilon(self, decay: float = 0.99, min_epsilon: float = 0.01) -> None:
        self.epsilon = max(float(min_epsilon), float(self.epsilon) * float(decay))

    def get_avg_entropy(self) -> float:
        """
        Strategy entropy:
        Convert Q-values to probabilities via softmax and compute entropy.
        Lower entropy => more confident / stabilized policy
        Higher entropy => more uncertain / exploring
        """
        entropies: List[float] = []
        for sid, action_vals in self.q_table.items():
            qs = np.array([action_vals["A"], action_vals["B"]], dtype=float)

            # Numerically stable softmax
            qs = qs - np.max(qs)
            exp_qs = np.exp(qs)
            probs = exp_qs / np.sum(exp_qs)

            entropies.append(float(scipy_entropy(probs)))

        return float(np.mean(entropies)) if entropies else 0.0


# =========================
# 4. Analysis utilities
# =========================
def calculate_diversity(actions_list: List[Action]) -> float:
    """
    Diversity score in [0, 1]:
      - 1.0 when A/B are balanced (50/50)
      - 0.0 when fully biased (all A or all B)
    """
    if not actions_list:
        return 0.0
    a_count = actions_list.count("A")
    ratio = a_count / len(actions_list)
    return float(1.0 - (2.0 * abs(0.5 - ratio)))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_simulation(
    culture_name: str,
    weights: Dict[str, float],
    episodes: int,
    custom_scenarios: List[Scenario],
    seed: int,
    learning_rate: float = 0.1,
    epsilon: float = 0.5,
) -> pd.DataFrame:
    set_global_seed(seed)

    agent = SimpleEGreedyAgent(
        name=culture_name,
        weights=weights,
        scenarios=custom_scenarios,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )

    history = {
        "episode": [],
        "reward": [],
        "diversity": [],
        "entropy": [],
    }

    progress = st.progress(0)
    for ep in range(int(episodes)):
        ep_actions: List[Action] = []
        ep_reward = 0.0

        for scn in custom_scenarios:
            action = agent.get_action(scn.sid)
            reward = agent.calculate_reward(scn.sid, action)
            agent.update(scn.sid, action, reward)

            ep_actions.append(action)
            ep_reward += reward

        agent.decay_epsilon()

        history["episode"].append(ep + 1)
        history["reward"].append(float(ep_reward))
        history["diversity"].append(calculate_diversity(ep_actions))
        history["entropy"].append(agent.get_avg_entropy())

        # Update progress ~ every 10 episodes (and last)
        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            progress.progress((ep + 1) / episodes)

    progress.empty()
    return pd.DataFrame(history)


# =========================
# 5. UI
# =========================
st.title("🎓 (TU Korea) Ethical AI Agent Reinforcement Learning Simulation")
st.markdown(
    """
This simulator demonstrates how an AI agent (using a simple epsilon-greedy RL approach)
learns to resolve ethical dilemmas under different cultural value weights.

1. **Environment setup**: define scenario rewards  
2. **Agent setup**: define cultural value weights  
3. **Analysis**: reward, diversity, and **strategy entropy**
"""
)

# --- Sidebar: Agent setup ---
st.sidebar.header("👤 2. Agent (Culture) Setup")

culture_presets = get_culture_presets()
selected_culture = st.sidebar.selectbox(
    "Culture preset",
    list(culture_presets.keys()),
    index=list(culture_presets.keys()).index("KOREA") if "KOREA" in culture_presets else 0,
)

episodes = st.sidebar.slider("Training episodes", 100, 1000, 300, step=50)
seed = st.sidebar.number_input("Random seed (reproducibility)", min_value=0, max_value=1_000_000, value=42, step=1)

st.sidebar.subheader("Value weight tuning")
mod_weights: Dict[str, float] = {}
culture_defaults = culture_presets[selected_culture]
for k in FRAMEWORKS:
    mod_weights[k] = st.sidebar.slider(k.capitalize(), 0.0, 1.0, float(culture_defaults.get(k, 0.0)))

final_weights = normalize_weights(mod_weights)

st.sidebar.markdown("---")
st.sidebar.caption("Normalized weights (sum to 1.0):")
st.sidebar.json(final_weights)

# --- Main: Environment setup ---
st.header("🌍 1. Environment Setup (Scenario Rewards)")
st.info(
    "Configure how each option is rewarded (-1.0 to 1.0) across the 4 ethical frameworks "
    "(Emotion, Social, Moral, Identity)."
)

DEFAULT_SCENARIOS = get_default_scenarios()
custom_scenarios: List[Scenario] = []

tabs = st.tabs([s.title for s in DEFAULT_SCENARIOS])
for i, (tab, default_scn) in enumerate(zip(tabs, DEFAULT_SCENARIOS)):
    with tab:
        st.markdown(f"> **Situation:** {default_scn.setup}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### 🅰 {default_scn.options['A']}")
            r_a = dict(default_scn.rewards["A"])
            for fw in FRAMEWORKS:
                r_a[fw] = st.slider(
                    f"[A] {fw}",
                    -1.0,
                    1.0,
                    float(r_a.get(fw, 0.0)),
                    0.1,
                    key=f"s{i}a_{fw}",
                )

        with col_b:
            st.markdown(f"### 🅱 {default_scn.options['B']}")
            r_b = dict(default_scn.rewards["B"])
            for fw in FRAMEWORKS:
                r_b[fw] = st.slider(
                    f"[B] {fw}",
                    -1.0,
                    1.0,
                    float(r_b.get(fw, 0.0)),
                    0.1,
                    key=f"s{i}b_{fw}",
                )

        custom_scenarios.append(
            Scenario(
                sid=default_scn.sid,
                title=default_scn.title,
                setup=default_scn.setup,
                options=default_scn.options,
                rewards={"A": r_a, "B": r_b},
            )
        )

# --- Run analysis ---
st.divider()
st.header("🚀 3. Simulation & Analysis")

if st.button("Run simulation", type="primary"):
    with st.spinner("Training the AI agent..."):
        df = run_simulation(
            culture_name=selected_culture,
            weights=final_weights,
            episodes=int(episodes),
            custom_scenarios=custom_scenarios,
            seed=int(seed),
        )

    st.success("Training complete!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📈 Total Reward")
        st.caption("Increases when the agent consistently selects actions aligned with its value weights.")
        st.line_chart(df, x="episode", y="reward")

    with col2:
        st.subheader("📉 Strategy Entropy")
        st.caption("Lower entropy indicates a more confident / stabilized strategy.")
        st.line_chart(df, x="episode", y="entropy")

    with col3:
        st.subheader("🔀 Action Diversity")
        st.caption("Closer to 1.0 indicates balanced exploration between A and B.")
        st.line_chart(df, x="episode", y="diversity")

    st.markdown("---")
    st.subheader("🔗 Correlation Analysis: Diversity vs Reward")

    r_val, p_val = pearsonr(df["diversity"], df["reward"])

    c_plot, c_stat = st.columns([2, 1])
    with c_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df["diversity"], df["reward"], alpha=0.65, linewidths=0.3)
        if len(df) > 1:
            z = np.polyfit(df["diversity"], df["reward"], 1)
            p = np.poly1d(z)
            xs = np.linspace(df["diversity"].min(), df["diversity"].max(), 100)
            ax.plot(xs, p(xs), "r--", label="Trend")
        ax.set_xlabel("Diversity (0=bias, 1=balanced)")
        ax.set_ylabel("Total Reward")
        ax.set_title(f"Diversity vs Reward (r={r_val:.2f})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with c_stat:
        st.metric("Pearson r", f"{r_val:.3f}")
        st.metric("p-value", f"{p_val:.3e}")
        if r_val > 0.3:
            st.success("✅ Positive correlation: higher diversity tends to increase reward.")
        elif r_val < -0.3:
            st.warning("⚠️ Negative correlation: lower diversity (more focus) tends to increase reward.")
        else:
            st.info("⏺ Little to no linear correlation detected.")

    with st.expander("📥 Download training log"):
        st.dataframe(df.head())
        st.download_button("Download CSV", df.to_csv(index=False), "ai_ethics_data.csv")