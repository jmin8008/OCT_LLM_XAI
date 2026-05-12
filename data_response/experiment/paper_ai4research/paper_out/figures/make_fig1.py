"""Publication-quality Figure 1: End-to-end autonomous research pipeline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── colour palette ─────────────────────────────────────────────────────────
NAVY      = "#1B2A4A"
BLUE      = "#2D5F8A"
TEAL      = "#1E6E7E"
ORANGE    = "#C8561E"
GRAY      = "#4A4A4A"
LIGHTGRAY = "#F2F4F7"
WHITE     = "#FFFFFF"
RED_DASH  = "#C0392B"

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis("off")
fig.patch.set_facecolor(WHITE)

# ── helper: rounded box ────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, color, alpha=1.0, radius=0.25, lw=1.5, ec=None):
    ec = ec or color
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=color, edgecolor=ec,
                       linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(p)

def txt(ax, x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontsize", 9)
    kw.setdefault("color", WHITE)
    kw.setdefault("zorder", 5)
    kw.setdefault("fontfamily", "DejaVu Sans")
    ax.text(x, y, s, **kw)

def arrow(ax, x0, y0, x1, y1, label="", color=GRAY, lw=2.0, head=0.25):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=f"->,head_width={head},head_length={head*0.8}",
                                color=color, lw=lw),
                zorder=4)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my+0.22, label, ha="center", va="bottom",
                fontsize=8, color=color, style="italic", zorder=5)

# ══════════════════════════════════════════════════════════════════════════
# COMPONENT 1 — OpenClaude Scientific Harness  (left)
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, 0.4, 1.2, 4.0, 4.6, NAVY, radius=0.3, lw=2)
txt(ax, 2.4, 5.35, "OpenClaude\nScientific Harness",
    fontsize=11, fontweight="bold", color=WHITE)

# LLM backend sub-box
rbox(ax, 0.75, 3.9, 3.3, 1.0, BLUE, radius=0.2, lw=1)
txt(ax, 2.4, 4.55, "LLM Backend", fontsize=8, color=WHITE, fontweight="bold")
txt(ax, 2.4, 4.18, "GLM-5.1-FP8  via  vLLM\n(4 × H100 80 GB)", fontsize=7.5, color="#C8D8EC")

# context.md sub-box
rbox(ax, 0.75, 2.65, 3.3, 1.05, TEAL, radius=0.2, lw=1)
txt(ax, 2.4, 3.30, "context.md", fontsize=8.5, color=WHITE, fontweight="bold", style="italic")
txt(ax, 2.4, 2.97, "Compounding domain memory\n(append-only; records insights +\nnegative results as constraints)",
    fontsize=7, color="#C8EAE8")

# program.md sub-box
rbox(ax, 0.75, 1.45, 3.3, 1.0, ORANGE, radius=0.2, lw=1, alpha=0.9)
txt(ax, 2.4, 2.08, "program.md", fontsize=8.5, color=WHITE, fontweight="bold", style="italic")
txt(ax, 2.4, 1.73, "Research constitution:\nNEVER STOP mandate +\nhard constraints",
    fontsize=7, color="#F8DCC8")

# ══════════════════════════════════════════════════════════════════════════
# COMPONENT 2 — AutoResearch Loop  (center)
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, 5.5, 1.2, 5.0, 4.6, NAVY, radius=0.3, lw=2)
txt(ax, 8.0, 5.35, "AutoResearch Loop",
    fontsize=11, fontweight="bold", color=WHITE)

# circular loop diagram
cx, cy, r = 8.0, 3.15, 1.15
theta = np.linspace(0, 2*np.pi, 300)
ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta),
        color="#7EB8D4", lw=1.5, zorder=4, alpha=0.6)

# four loop nodes
nodes = [
    (0.5*np.pi,  "Hypothesize", "#3A7DC9"),
    (0*np.pi,    "Implement\ntrain.py", "#2E8B57"),
    (-0.5*np.pi, "Execute\n900 s budget", "#8B5E2E"),
    (np.pi,      "Analyze", "#6A3C8B"),
]
for angle, label, c in nodes:
    nx = cx + r * np.cos(angle)
    ny = cy + r * np.sin(angle)
    rbox(ax, nx-0.55, ny-0.30, 1.1, 0.60, c, radius=0.15, lw=1)
    txt(ax, nx, ny, label, fontsize=7, color=WHITE)

# loop arrows (small curved arrows between nodes)
for i, (angle, _, _) in enumerate(nodes):
    a0 = angle
    a1 = nodes[(i+1) % len(nodes)][0]
    mid = (a0 + a1) / 2
    sx = cx + (r+0.08) * np.cos(a0 - 0.35)
    sy = cy + (r+0.08) * np.sin(a0 - 0.35)
    ex = cx + (r+0.08) * np.cos(a1 + 0.35)
    ey = cy + (r+0.08) * np.sin(a1 + 0.35)
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.10",
                                color="#7EB8D4", lw=1.2,
                                connectionstyle=f"arc3,rad=-0.4"),
                zorder=4)

# stats label inside loop
txt(ax, cx, cy, "99+ exps\n4 sessions\n~30 h total",
    fontsize=7, color="#B0D0E8", ha="center")

# artifacts box at bottom of component 2
rbox(ax, 5.85, 1.38, 4.3, 0.7, BLUE, radius=0.15, lw=1)
txt(ax, 8.0, 1.73, "Artifacts: results.json  ·  changelog.md  ·  archive/train_expN.py",
    fontsize=7, color="#C8D8EC")

# dataset box (below center component)
rbox(ax, 6.35, 0.15, 3.3, 0.75, GRAY, radius=0.2, lw=1.5, ec="#AAAAAA")
txt(ax, 8.0, 0.52, "APTOS 2021 OCT Dataset\n221 patients · 2,875 images · 3 tasks",
    fontsize=7.5, color=WHITE)
arrow(ax, 8.0, 0.9, 8.0, 1.37, color="#AAAAAA", lw=1.5, head=0.15)

# ══════════════════════════════════════════════════════════════════════════
# COMPONENT 3 — Paper Orchestra  (right)
# ══════════════════════════════════════════════════════════════════════════
rbox(ax, 11.6, 1.2, 4.0, 4.6, NAVY, radius=0.3, lw=2)
txt(ax, 13.6, 5.35, "Paper Orchestra",
    fontsize=11, fontweight="bold", color=WHITE)

stages = [
    ("① Outliner",             "#3A7DC9", 4.75),
    ("② Lit Review + Figures", "#2E8B57", 4.05),
    ("③ Section Writer",       "#8B5E2E", 3.35),
    ("④ Reviewer Simulation",  "#6A3C8B", 2.65),
    ("⑤ Refiner Loop",         "#5E6E2E", 1.95),
]
for label, c, ypos in stages:
    rbox(ax, 11.95, ypos-0.22, 3.3, 0.48, c, radius=0.12, lw=1)
    txt(ax, 13.6, ypos+0.02, label, fontsize=8, color=WHITE)

# output box
rbox(ax, 11.95, 1.38, 3.3, 0.38, TEAL, radius=0.12, lw=1)
txt(ax, 13.6, 1.57, "→  paper.pdf   (45 min wall time)", fontsize=7.5, color=WHITE)

# submission box (far right)
rbox(ax, 12.1, 0.15, 3.0, 0.75, RED_DASH, radius=0.2, lw=1.5)
txt(ax, 13.6, 0.52, "Submission-Ready\nManuscript (this paper)",
    fontsize=7.5, color=WHITE, fontweight="bold")
arrow(ax, 13.6, 1.37, 13.6, 0.9, color=RED_DASH, lw=1.5, head=0.15)

# ══════════════════════════════════════════════════════════════════════════
# INTER-COMPONENT ARROWS
# ══════════════════════════════════════════════════════════════════════════
# Harness ↔ AutoResearch (bidirectional)
arrow(ax, 4.4,  3.80, 5.5,  3.80, label="Structured Context", color="#7EB8D4", lw=2.0, head=0.20)
arrow(ax, 5.5,  3.30, 4.4,  3.30, color="#7EB8D4", lw=2.0, head=0.20)

# AutoResearch → Paper Orchestra
arrow(ax, 10.5, 3.15, 11.6, 3.15, label="Research Artifacts", color="#A8C8A0", lw=2.0, head=0.20)

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
ax.text(8.0, 6.65, "End-to-End Autonomous ML Research Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=NAVY, zorder=5)

plt.tight_layout(pad=0.3)
plt.savefig("fig_pipeline_overview.png", dpi=180, bbox_inches="tight",
            facecolor=WHITE, edgecolor="none")
print("Saved fig_pipeline_overview.png")
