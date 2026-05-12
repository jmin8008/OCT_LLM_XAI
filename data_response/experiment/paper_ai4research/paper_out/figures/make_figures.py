"""
Generate two figures for the ICML Workshop paper on autonomous ML research pipeline.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# FIGURE 1: Pipeline overview
# ─────────────────────────────────────────────────────────────

def make_pipeline_figure():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Color palette
    C_DARK   = '#1a2b4a'   # dark navy
    C_MID    = '#2e5d9f'   # medium blue
    C_LIGHT  = '#dce8f8'   # light blue fill
    C_ACCENT = '#4a90d9'   # accent blue
    C_GRAY   = '#6b7a8d'   # gray
    C_LGRAY  = '#f0f4f8'   # light gray fill
    C_GREEN  = '#2e7d32'   # dataset green
    C_LGREEN = '#e8f5e9'   # light green

    def rounded_box(ax, x, y, w, h, text_lines, color_face, color_edge,
                    title=None, title_color='white', title_bg=None,
                    fontsize=10, title_fontsize=11):
        """Draw a rounded box with optional title bar."""
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.08",
                             linewidth=1.8,
                             edgecolor=color_edge,
                             facecolor=color_face,
                             zorder=2)
        ax.add_patch(box)

        if title is not None:
            tbg = title_bg if title_bg else color_edge
            title_bar = FancyBboxPatch((x, y + h - 0.72), w, 0.72,
                                       boxstyle="round,pad=0.05",
                                       linewidth=0,
                                       edgecolor=tbg,
                                       facecolor=tbg,
                                       zorder=3)
            ax.add_patch(title_bar)
            ax.text(x + w/2, y + h - 0.36, title,
                    ha='center', va='center', fontsize=title_fontsize,
                    fontweight='bold', color=title_color, zorder=4)

        # Body text
        n = len(text_lines)
        body_h = h - (0.72 if title else 0)
        body_y_top = y + (h - 0.72 if title else h)
        for i, line in enumerate(text_lines):
            ty = body_y_top - body_h * (i + 0.55) / n
            ax.text(x + w/2, ty, line,
                    ha='center', va='center', fontsize=fontsize,
                    color=C_DARK, zorder=4)

    def arrow(ax, x1, y1, x2, y2, label='', bidirectional=False, color=C_MID):
        style = 'simple,tail_width=2.5,head_width=12,head_length=8'
        ax.annotate('',
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style,
                                   color=color,
                                   lw=0),
                    zorder=5)
        if bidirectional:
            ax.annotate('',
                        xy=(x1, y1), xytext=(x2, y2),
                        arrowprops=dict(arrowstyle=style,
                                       color=color,
                                       lw=0),
                        zorder=5)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.28, label, ha='center', va='bottom',
                    fontsize=9, color=C_GRAY, fontstyle='italic', zorder=6)

    # ── LEFT BOX: OpenClaude Science Agent Harness ──
    bx1, by1, bw, bh = 0.5, 2.8, 3.8, 3.8
    rounded_box(ax, bx1, by1, bw, bh,
                text_lines=[
                    'Qwen3-27B-A22B',
                    '',
                    'CLAUDE.md',
                    '(domain memory)',
                    '',
                    'program.md',
                    '(research constitution)',
                ],
                color_face=C_LIGHT, color_edge=C_MID,
                title='OpenClaude Science Agent Harness',
                title_fontsize=9.5, fontsize=9.5)

    # ── CENTER BOX: AutoResearch Loop ──
    bx2, by2, bw2, bh2 = 5.7, 2.8, 4.6, 3.8
    rounded_box(ax, bx2, by2, bw2, bh2,
                text_lines=[],
                color_face=C_LGRAY, color_edge=C_DARK,
                title='AutoResearch Loop',
                title_fontsize=10.5)

    # Circular arrow inside center box
    theta = np.linspace(0.18 * np.pi, 1.82 * np.pi, 200)
    cx, cy, r = bx2 + bw2/2, by2 + bh2/2 - 0.25, 1.02
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
            color=C_ACCENT, linewidth=2.2, zorder=4)
    # Arrowhead at the end of arc
    end_x = cx + r * np.cos(1.82 * np.pi)
    end_y = cy + r * np.sin(1.82 * np.pi)
    ax.annotate('',
                xy=(end_x - 0.01, end_y + 0.18),
                xytext=(end_x, end_y),
                arrowprops=dict(arrowstyle='->', color=C_ACCENT,
                                lw=2, mutation_scale=18),
                zorder=5)

    # Labels on the circular loop
    loop_labels = [
        (cx,          cy + r + 0.22,  'Hypothesis',   'top'),
        (cx + r + 0.1, cy,            'Implement',    'left'),
        (cx,          cy - r - 0.22,  'Execute',      'bottom'),
        (cx - r - 0.1, cy,            'Analyze',      'right'),
    ]
    for lx, ly, ltext, va in loop_labels:
        ax.text(lx, ly, ltext, ha='center', va='center',
                fontsize=9.5, fontweight='semibold', color=C_DARK, zorder=6)

    # ── RIGHT BOX: Paper Orchestra ──
    bx3, by3, bw3, bh3 = 11.7, 2.8, 3.8, 3.8
    rounded_box(ax, bx3, by3, bw3, bh3,
                text_lines=[
                    'Outline',
                    'Lit Review',
                    '',
                    'Figures  →  Section Writing',
                    '',
                    'Review + Refine',
                    '',
                    'paper.pdf',
                ],
                color_face=C_LIGHT, color_edge=C_MID,
                title='Paper Orchestra',
                title_fontsize=10.5, fontsize=9.2)

    # ── DATASET BOX: APTOS 2021 OCT ──
    dx, dy, dw, dh = 5.3, 0.5, 5.4, 1.0
    dataset_box = FancyBboxPatch((dx, dy), dw, dh,
                                  boxstyle="round,pad=0.06",
                                  linewidth=1.5,
                                  edgecolor=C_GREEN,
                                  facecolor=C_LGREEN,
                                  zorder=2)
    ax.add_patch(dataset_box)
    ax.text(dx + dw/2, dy + dh/2, 'APTOS 2021 OCT Dataset',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=C_GREEN, zorder=4)

    # ── OUTPUT BOX: Submission-Ready Paper ──
    ox, oy, ow, oh = 13.05, 7.15, 2.4, 0.85
    out_box = FancyBboxPatch((ox, oy), ow, oh,
                              boxstyle="round,pad=0.06",
                              linewidth=1.5,
                              edgecolor='#b71c1c',
                              facecolor='#ffebee',
                              zorder=2)
    ax.add_patch(out_box)
    ax.text(ox + ow/2, oy + oh/2, 'Submission-Ready\nPaper',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='#b71c1c', zorder=4)

    # ── ARROWS ──
    # LEFT → CENTER bidirectional
    arrow(ax, bx1 + bw, by1 + bh/2, bx2, by2 + bh2/2,
          label='Structured Context', bidirectional=True)

    # CENTER → RIGHT
    arrow(ax, bx2 + bw2, by2 + bh2/2, bx3, by3 + bh3/2,
          label='Research Artifacts', bidirectional=False)

    # RIGHT → Output box
    arrow(ax, bx3 + bw3/2, by3 + bh3, ox + ow/2, oy,
          bidirectional=False, color='#b71c1c')

    # Dataset → CENTER (upward)
    ax.annotate('',
                xy=(bx2 + bw2/2, by2),
                xytext=(dx + dw/2, dy + dh),
                arrowprops=dict(arrowstyle='->', color=C_GREEN,
                                lw=2, mutation_scale=16),
                zorder=5)

    # ── TITLE ──
    ax.text(8, 8.6, 'Autonomous ML Research Pipeline',
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=C_DARK)

    out_path = os.path.join(OUT_DIR, 'fig_pipeline_overview.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}  ({os.path.getsize(out_path):,} bytes)")


# ─────────────────────────────────────────────────────────────
# FIGURE 2: Task 1 experiment progression
# ─────────────────────────────────────────────────────────────

def make_progression_figure():
    # Data
    keep_pts = {
        0: 0.9400, 4: 0.9418, 5: 0.9475, 12: 0.9491,
        21: 0.9512, 22: 0.9576, 25: 0.9612, 27: 0.9607,
        30: 0.9630, 32: 0.9636,
    }
    revert_pts = {
        1: 0.9408, 2: 0.9247, 3: 0.9285, 6: 0.9467,
        7: 0.9444, 8: 0.9413, 9: 0.9326, 10: 0.9461,
        11: 0.9268, 13: 0.9356, 14: 0.9456, 15: 0.9489,
        16: 0.9451, 17: 0.9443, 18: 0.9460, 19: 0.9434,
        20: 0.9441, 23: 0.9551, 24: 0.9563, 26: 0.9598,
        28: 0.9589, 29: 0.9598, 31: 0.9614,
    }

    # Phases: (label, x_start, x_end)
    phases = [
        ('Phase 1\n(0–3)',   0,  3),
        ('Phase 2\n(4–5)',   4,  5),
        ('Phase 3\n(6–13)',  6, 13),
        ('Phase 4\n(14–21)', 14, 21),
        ('Phase 5\n(22–32)', 22, 32),
    ]
    phase_colors = ['#e3f2fd', '#f3e5f5', '#e8f5e9', '#fff3e0', '#fce4ec']

    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor('white')

    # Phase shading (in the plot area)
    for (label, xs, xe), pc in zip(phases, phase_colors):
        ax.axvspan(xs - 0.5, xe + 0.5, alpha=0.35, color=pc, zorder=0)

    # BlueSky baseline
    ax.axhline(y=0.9225, color='#c62828', linewidth=1.6,
               linestyle='--', zorder=2, label='BlueSky (competition winner)')
    ax.text(32.4, 0.9225, 'BlueSky\n(0.9225)', color='#c62828',
            va='center', fontsize=8.5, fontweight='bold')

    # Connecting line through KEEP points (chronological order)
    kx = sorted(keep_pts.keys())
    ky = [keep_pts[x] for x in kx]
    ax.plot(kx, ky, color='#1565c0', linewidth=1.8,
            linestyle='-', alpha=0.7, zorder=3)

    # REVERT scatter
    rx = list(revert_pts.keys())
    ry = [revert_pts[x] for x in rx]
    ax.scatter(rx, ry, marker='x', s=55, linewidths=1.6,
               color='#90a4ae', zorder=4, label='REVERT (not kept)')

    # KEEP scatter
    ax.scatter(kx, ky, marker='o', s=70, color='#1565c0',
               edgecolors='#0d47a1', linewidths=1.0,
               zorder=5, label='KEEP (accepted)')

    # Annotation for final experiment
    ax.annotate('Final: 0.9636',
                xy=(32, 0.9636),
                xytext=(28.5, 0.9670),
                fontsize=9.5, fontweight='bold', color='#1a237e',
                arrowprops=dict(arrowstyle='->', color='#1a237e',
                                lw=1.5, mutation_scale=14),
                zorder=6)

    # Phase labels below x-axis
    for (label, xs, xe), pc in zip(phases, phase_colors):
        mx = (xs + xe) / 2
        ax.text(mx, 0.899, label, ha='center', va='top',
                fontsize=8, color='#555555',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=pc,
                          edgecolor='#aaaaaa', linewidth=0.7))

    # Axes formatting
    ax.set_xlim(-1, 34)
    ax.set_ylim(0.900, 0.975)
    ax.set_xlabel('Experiment Number', fontsize=12, labelpad=16)
    ax.set_ylabel('Mean AUC', fontsize=12)
    ax.set_title('Task 1: Mean AUC Progression Across 32 Experiments\n'
                 '(AutoResearch Loop — APTOS 2021 OCT)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(range(0, 33, 2))
    ax.set_yticks(np.arange(0.90, 0.976, 0.005))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.3f}'))
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=1)
    ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.3, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9)

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = os.path.join(OUT_DIR, 'fig_task1_experiment_progression.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}  ({os.path.getsize(out_path):,} bytes)")


if __name__ == '__main__':
    make_pipeline_figure()
    make_progression_figure()
    print("Done.")
