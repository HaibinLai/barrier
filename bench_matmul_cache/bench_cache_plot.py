#!/usr/bin/env python3
"""
bench_cache_plot.py — Plot matmul kernel cache staircase effect

Reads:  bench_matmul_cache/results/bench_cache.csv
Output: pdf_images/matmul_cache_staircase.pdf

X-axis: Working set size (KB, log scale)
Y-axis: Throughput (GFLOPS) — primary;  Latency per row (ns) — secondary
Vertical dashed lines at L1d (32KB) and L2 (1MB) boundaries.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os
import sys

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(SCRIPT_DIR, "results", "bench_cache.csv")
OUTPUT_DIR = os.path.join(REPO_DIR, "pdf_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cache sizes (Intel Xeon Platinum 8160) ───────────────────────────
L1D_KB = 32       # 32 KB per core
L2_KB = 1024      # 1 MB per core
# LLC_KB = 33792  # 33 MB shared (less useful for single-core plot)

# ── Style ────────────────────────────────────────────────────────────
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

COLOR_GFLOPS   = '#1f77b4'   # blue
COLOR_NS_ROW   = '#ff7f0e'   # orange
COLOR_L1       = '#d62728'   # red
COLOR_L2       = '#2ca02c'   # green
COLOR_BG_L1    = '#d6272810' # translucent
COLOR_BG_L2    = '#ff7f0e10'


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        print(f"Run bench_matmul_cache/bench_cache_run.sh first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} data points from {CSV_PATH}")
    print(df.to_string(index=False))
    print()

    ws = df['working_set_kb'].values
    gflops = df['gflops'].values
    ns_per_row = df['ns_per_row'].values
    b0 = df['b0'].values

    # ── Figure: single panel, GFLOPS only (cleaner for paper) ──────────
    fig, ax1 = plt.subplots(1, 1, figsize=(5.5, 3.8))

    # Primary Y-axis: GFLOPS
    ax1.semilogx(ws, gflops, 'o-', color=COLOR_GFLOPS, linewidth=2.2,
                 markersize=5, label='vec\\_dot\\_f16 throughput', zorder=5)
    ax1.set_xlabel('Tile Working Set Size (KB)')
    ax1.set_ylabel('Throughput (GFLOPS)')

    # ── Cache boundary lines ─────────────────────────────────────────
    ax1.axvline(x=L1D_KB, color=COLOR_L1, linestyle='--', linewidth=1.5,
                alpha=0.8, zorder=3)
    ax1.axvline(x=L2_KB, color=COLOR_L2, linestyle='--', linewidth=1.5,
                alpha=0.8, zorder=3)

    # Label the cache boundaries at the top
    trans = ax1.get_xaxis_transform()
    ax1.text(L1D_KB, 1.03, 'L1d (32 KB)', transform=trans,
             fontsize=8, color=COLOR_L1, ha='center', va='bottom',
             fontweight='bold')
    ax1.text(L2_KB, 1.03, 'L2 (1 MB)', transform=trans,
             fontsize=8, color=COLOR_L2, ha='center', va='bottom',
             fontweight='bold')

    # ── Shaded cache regions ─────────────────────────────────────────
    ax1.axvspan(ws.min() * 0.8, L1D_KB, alpha=0.06, color=COLOR_L1,
                zorder=1, label='_nolegend_')
    ax1.axvspan(L1D_KB, L2_KB, alpha=0.05, color=COLOR_L2,
                zorder=1, label='_nolegend_')

    # ── Key annotations: plateau and cliff ───────────────────────────
    # Find peak GFLOPS (the plateau)
    peak_gf = gflops.max()
    bottom_gf = gflops[ws > L2_KB].min() if any(ws > L2_KB) else gflops[-1]
    speedup = peak_gf / bottom_gf

    # Annotate the plateau region
    plateau_idx = np.argmax(gflops)
    ax1.annotate(f'{peak_gf:.1f} GFLOPS\n(fits in L2)',
                 xy=(ws[plateau_idx], gflops[plateau_idx]),
                 xytext=(ws[plateau_idx] * 2.5, gflops[plateau_idx] - 8),
                 fontsize=8, color=COLOR_GFLOPS,
                 arrowprops=dict(arrowstyle='->', color=COLOR_GFLOPS, lw=1.2),
                 ha='left', fontweight='bold')

    # Annotate the bottom (after L2 miss)
    bottom_idx = len(gflops) - 1
    ax1.annotate(f'{bottom_gf:.1f} GFLOPS\n(L2 miss)',
                 xy=(ws[bottom_idx], gflops[bottom_idx]),
                 xytext=(ws[bottom_idx] * 0.25, gflops[bottom_idx] + 5),
                 fontsize=8, color='#666666',
                 arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2),
                 ha='center')

    # Speedup annotation between the two
    mid_x = np.sqrt(ws[plateau_idx] * ws[bottom_idx])  # geometric mean
    ax1.annotate('',
                 xy=(mid_x, bottom_gf + 1), xytext=(mid_x, peak_gf - 1),
                 arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
    ax1.text(mid_x * 1.15, (peak_gf + bottom_gf) / 2,
             f'{speedup:.1f}$\\times$',
             fontsize=10, fontweight='bold', color='#333333',
             ha='left', va='center')

    # ── Legend ────────────────────────────────────────────────────────
    ax1.legend(loc='lower left', framealpha=0.9)

    # ── Grid ─────────────────────────────────────────────────────────
    ax1.grid(True, which='both', linestyle=':', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.set_ylim(bottom=0)

    # ── X-axis tick formatting ───────────────────────────────────────
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:.0f}' if x < 1024 else f'{x/1024:.0f} MB'))

    # ── Title ────────────────────────────────────────────────────────
    ax1.set_title(
        'Matmul Kernel Cache Effect: Tile Size $b_0$ Sweep\n'
        'FP16 vec\\_dot, K=2560 (Qwen3-4B FFN), Intel Xeon 8160',
        fontsize=10, fontweight='bold', pad=18)

    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────
    pdf_path = os.path.join(OUTPUT_DIR, "matmul_cache_staircase.pdf")
    png_path = pdf_path.replace('.pdf', '.png')

    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {pdf_path}")

    fig.savefig(png_path, bbox_inches='tight', dpi=180)
    print(f"Saved: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
