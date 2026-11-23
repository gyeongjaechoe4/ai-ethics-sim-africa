"""Draws the simulator structure diagram (workshop deliverable 1). Needs a Korean font (NanumGothic)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
OUT = Path(__file__).resolve().parent / "out" / "fig_simulator_structure.png"

fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")


def box(x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=13, weight="bold", color=ec)
    for i, ln in enumerate(lines):
        ax.text(x + 0.18, y + h - 0.8 - i * 0.42, ln, ha="left", va="center", fontsize=10)


def arrow(x1, y1, x2, y2, txt=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#404040"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, txt, ha="center", fontsize=9, color="#404040")


ax.text(7, 6.9, "윤리 강화학습 시뮬레이터 구조도 (app_reinforce.py · 팀 AFRICA)", ha="center", fontsize=15, weight="bold")

box(0.2, 1.3, 3.9, 5.1, "① 입력 (실행 전에 정함)", [
    "문화 가중치 (합=1로 정규화)",
    "  AFRICA 최적: E.15 S.40 M.25 I.20",
    "시나리오 3개 × A/B 보상 벡터",
    "  (default-v1, 각 값 -1~+1)",
    "에피소드 수: 300",
    "난수 시드: 42 (+ 123·2026·7777·31337)",
    "학습률 α = 0.1",
    "탐색률 ε₀ = 0.5, decay 0.99, 최소 0.01",
    "평가 구간: 에피소드 251~300",
], "#eaf1fb", "#1f4e79")

box(5.0, 1.3, 4.0, 5.1, "② 규칙 (학습 과정)", [
    "에피소드마다 S1→S2→S3 순서로:",
    "1. ε-greedy 행동 선택",
    "   확률 ε: A/B 무작위 (탐색)",
    "   나머지: Q값 큰 쪽 (활용)",
    "2. 보상 R = 10 × Σ(가중치 × 보상값)",
    "3. Q 갱신: Q ← Q + α(R − Q)",
    "에피소드 끝: ε ← max(0.01, ε×0.99)",
    "엔트로피: Q → softmax 확률 → 엔트로피",
], "#fff4e5", "#c55a11")

box(9.9, 1.3, 3.9, 5.1, "③ 출력 (실행 후 측정)", [
    "에피소드별 총보상 곡선",
    "전략 엔트로피 (낮을수록 확신)",
    "행동 다양성",
    "  1 = A/B 반반, 0 = 한쪽만",
    "A/B 선택 비율",
    "다양성-보상 Pearson r",
    "실행 로그 CSV",
    "  (episode, reward, diversity,",
    "   entropy, actions)",
], "#e9f5e1", "#38761d")

arrow(4.15, 3.85, 4.95, 3.85, "넣는다")
arrow(9.05, 3.85, 9.85, 3.85, "기록한다")
ax.annotate("", xy=(6.2, 1.25), xytext=(7.8, 1.25),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#c55a11", connectionstyle="arc3,rad=-0.6"))
ax.text(7.0, 0.55, "에피소드 300번 반복 (Q값·ε가 다음 선택에 영향)", ha="center", fontsize=9.5, color="#c55a11")
ax.text(0.2, 0.15, "실험에서 바꾸는 것(요인)은 ①의 한 칸뿐, ②는 코드로 고정, ③은 같은 평가 구간에서 잰다.",
        fontsize=10, color="#404040")
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(OUT)
