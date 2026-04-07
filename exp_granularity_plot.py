#!/usr/bin/env python3
"""
exp_granularity_plot.py — Matmul Task Granularity Sensitivity Figure

Generates a 2-panel figure for paper Section 4.3 (fig:cache-effect):
  Panel (a): Single-core tile size vs GFLOPS (cache effect)
  Panel (b): Multi-core task count vs throughput (work-stealing effect)

Input:  exp_granularity_results/panel_a_single_core.csv
        exp_granularity_results/panel_b_multi_core.csv
Output: pdf_images/task_granularity_sensitivity.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os
import sys

# ── Configuration ─────────────────────────────────────────────────────
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_granularity_results")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matmul shape for GFLOPS conversion
# Qwen3-4B FFN up: dst = src1 @ src0^T, where src0 is 6912x2560, src1 is 128x2560
# FLOPs per matmul = 2 * M * K * N = 2 * 6912 * 2560 * 128
# But llama-bench reports overall pp t/s (tokens/s), not per-matmul.
# We'll plot t/s directly (more meaningful for the reader) and optionally GFLOPS.
M = 6912
K = 2560
NR1 = 128

# Cache sizes (Intel Xeon Platinum 8160)
L1D_KB = 32
L2_KB = 1024  # 1 MB per core
LLC_KB = 33 * 1024  # 33 MB per socket (shared)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

COLOR_TF = '#2ca02c'     # green — Taskflow/work-stealing
COLOR_FJ = '#1f77b4'     # blue — Fork-Join
COLOR_L1 = '#d62728'     # red
COLOR_L2 = '#ff7f0e'     # orange
COLOR_LLC = '#9467bd'    # purple


def plot_panel_a(ax, csv_path):
    """Panel (a): Single-core tile size vs pp throughput."""
    df = pd.read_csv(csv_path)
    df = df[df['pp_ts'].astype(float) > 0].copy()
    df['working_set_kb'] = df['working_set_kb'].astype(float)
    df['pp_ts'] = df['pp_ts'].astype(float)

    ax.semilogx(df['working_set_kb'], df['pp_ts'],
                'o-', color=COLOR_TF, linewidth=2, markersize=6,
                label='Single-core throughput', zorder=5)

    # Cache boundary lines
    ax.axvline(x=L1D_KB, color=COLOR_L1, linestyle='--', linewidth=1.2,
               alpha=0.8, label=f'L1d ({L1D_KB} KB)')
    ax.axvline(x=L2_KB, color=COLOR_L2, linestyle='--', linewidth=1.2,
               alpha=0.8, label=f'L2 ({L2_KB} KB)')

    ax.set_xlabel('Tile Working Set Size (KB)')
    ax.set_ylabel('Prompt Processing (tokens/s)')
    ax.set_title('(a) Tile Size vs Single-Core Performance')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)

    # X-axis formatting
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:.0f}' if x < 1024 else f'{x/1024:.0f}M'))

    # Annotate cache regions
    ymin, ymax = ax.get_ylim()
    ymid = (ymin + ymax) / 2
    ax.annotate('L1d\nfits', xy=(L1D_KB / 3, ymid),
                fontsize=7, color=COLOR_L1, alpha=0.6, ha='center')


def plot_panel_b(ax, csv_path):
    """Panel (b): Multi-core task count vs throughput (Taskflow vs FJ)."""
    df = pd.read_csv(csv_path)
    df['pp_ts'] = df['pp_ts'].astype(float)
    df['total_tasks'] = df['total_tasks'].astype(int)

    df_tf = df[df['mode'] == 'taskflow'].sort_values('total_tasks')
    df_fj = df[df['mode'] == 'forkjoin'].sort_values('total_tasks')

    # Taskflow line
    ax.semilogx(df_tf['total_tasks'], df_tf['pp_ts'],
                'D-', color=COLOR_TF, linewidth=2, markersize=6,
                label='TaskInfer (work-stealing)', zorder=5)

    # Fork-Join line
    ax.semilogx(df_fj['total_tasks'], df_fj['pp_ts'],
                's-', color=COLOR_FJ, linewidth=2, markersize=6,
                label='Fork-Join (static)', zorder=5)

    # Vertical line at 24 cores
    ax.axvline(x=24, color='gray', linestyle=':', linewidth=1,
               alpha=0.6, label='= 24 cores')

    # Find and annotate peak
    if len(df_tf) > 0:
        peak_idx = df_tf['pp_ts'].idxmax()
        peak_x = df_tf.loc[peak_idx, 'total_tasks']
        peak_y = df_tf.loc[peak_idx, 'pp_ts']
        ax.annotate(f'peak: {peak_y:.1f} t/s\n({peak_x} tasks)',
                    xy=(peak_x, peak_y),
                    xytext=(peak_x * 2, peak_y * 0.92),
                    fontsize=8, color=COLOR_TF,
                    arrowprops=dict(arrowstyle='->', color=COLOR_TF, lw=1),
                    ha='left')

    ax.set_xlabel('Number of Tasks (= # chunks)')
    ax.set_ylabel('Prompt Processing (tokens/s, 24 cores)')
    ax.set_title('(b) Task Granularity vs Multi-Core Throughput')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    ax.set_axisbelow(True)

    # X-axis: integer task counts
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def main():
    csv_a = os.path.join(RESULT_DIR, "panel_a_single_core.csv")
    csv_b = os.path.join(RESULT_DIR, "panel_b_multi_core.csv")

    if not os.path.exists(csv_a):
        print(f"ERROR: {csv_a} not found. Run exp_granularity_run.sh first.")
        sys.exit(1)
    if not os.path.exists(csv_b):
        print(f"ERROR: {csv_b} not found. Run exp_granularity_run.sh first.")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    plot_panel_a(ax1, csv_a)
    plot_panel_b(ax2, csv_b)

    fig.suptitle('Matmul Task Granularity Sensitivity — Qwen3-4B FFN up (6912$\\times$2560)\n'
                 'Intel Xeon Platinum 8160, 1 NUMA node, FP16',
                 fontsize=12, fontweight='bold', y=1.02)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = os.path.join(OUTPUT_DIR, "task_granularity_sensitivity.pdf")
    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    print(f"Saved: {outpath}")

    # Also save PNG for quick preview
    outpath_png = outpath.replace('.pdf', '.png')
    fig.savefig(outpath_png, bbox_inches='tight', dpi=180)
    print(f"Saved: {outpath_png}")


if __name__ == "__main__":
    main()
