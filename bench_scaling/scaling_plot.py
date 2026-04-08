#!/usr/bin/env python3
"""
scaling_plot.py — Generate scaling experiment figures for the paper.

Generates 6 figures:
  1. Strong Scaling curve (throughput vs NUMA nodes)
  2. Scaling Efficiency E(N) = T(N) / (N * T(1))
  3. Barrier Amplification A(N) vs Compute Amplification C(N)
  4. Thread Load Imbalance (violin/box plot)
  5. Thread Sweep (8N: throughput, barrier%, idle% vs thread count)
  6. Per-op Barrier contribution (if per-op data available)

Usage:
    python3 scaling_plot.py [--summary scaling_summary.csv] [--thread-data scaling_per_thread.csv]
    python3 scaling_plot.py --summary scaling_summary.csv --thread-data scaling_per_thread.csv --output-dir pdf_images
"""

import argparse
import csv
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ============================================================================
# Style Configuration
# ============================================================================
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# NUMA domain colors (up to 8 domains)
NUMA_COLORS = ["#4285F4", "#EA4335", "#34A853", "#FBBC04",
               "#9C27B0", "#FF6D00", "#00BCD4", "#795548"]

# Line/bar colors
COLOR_TG = "#d62728"       # red — decode/TG
COLOR_PP = "#1f77b4"       # blue — prefill/PP
COLOR_BARRIER = "#ff7f0e"  # orange — barrier
COLOR_COMPUTE = "#2ca02c"  # green — compute
COLOR_IDLE = "#9467bd"     # purple — idle
COLOR_IDEAL = "#888888"    # gray — ideal scaling


def load_csv(filepath):
    """Load CSV into list of dicts."""
    with open(filepath) as f:
        return list(csv.DictReader(f))


def filter_data(data, **kwargs):
    """Filter data rows by column values."""
    result = []
    for row in data:
        match = True
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple)):
                if row.get(k) not in [str(x) for x in v]:
                    match = False
                    break
            else:
                if row.get(k) != str(v):
                    match = False
                    break
        if match:
            result.append(row)
    return result


def get_val(data, numa, metric, model="qwen4b", system="fj", stage="decode", threads=None):
    """Get a metric value for a specific config.

    If threads is None, use the NUMA-scaling convention: 24 threads per NUMA node.
    """
    if threads is None:
        numa_count = int(numa[:-1])
        threads = numa_count * 24

    rows = filter_data(data, model=model, numa=numa, system=system, stage=stage, threads=threads)
    if not rows:
        # Try alternate model names
        for alt in ["qwen3_4b", "qwen4b"]:
            rows = filter_data(data, model=alt, numa=numa, system=system, stage=stage, threads=threads)
            if rows:
                break
    if rows:
        return float(rows[0].get(metric, 0))
    return 0.0


# ============================================================================
# Figure 1: Strong Scaling Curve
# ============================================================================
def plot_strong_scaling(data, output_path, model="qwen4b"):
    """Throughput vs NUMA nodes — with ideal linear scaling reference."""
    numas = ["1n", "4n", "8n"]
    numa_count = [1, 4, 8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- TG (decode) ---
    tg_vals = [get_val(data, n, "speed_tg", model=model) for n in numas]
    if tg_vals[0] > 0:
        tg_ideal = [tg_vals[0] * n for n in numa_count]
    else:
        tg_ideal = [0] * len(numa_count)

    ax1.plot(numa_count, tg_vals, "o-", color=COLOR_TG, linewidth=2.5, markersize=8,
             label="FJ Pure (actual)", zorder=5)
    ax1.plot(numa_count, tg_ideal, "--", color=COLOR_IDEAL, linewidth=1.5,
             label="Ideal linear", alpha=0.7)

    for i, (n, v) in enumerate(zip(numa_count, tg_vals)):
        ax1.annotate(f"{v:.2f}", (n, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color=COLOR_TG)

    ax1.set_xlabel("NUMA Nodes", fontsize=11)
    ax1.set_ylabel("Decode Throughput (tokens/s)", fontsize=11)
    ax1.set_title("(a) Decode (TG) Strong Scaling", fontsize=12, fontweight="bold")
    ax1.set_xticks(numa_count)
    ax1.legend(fontsize=9)
    ax1.grid(linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    # --- PP (prefill) ---
    pp_vals = [get_val(data, n, "speed_pp", model=model) for n in numas]
    if pp_vals[0] > 0:
        pp_ideal = [pp_vals[0] * n for n in numa_count]
    else:
        pp_ideal = [0] * len(numa_count)

    ax2.plot(numa_count, pp_vals, "s-", color=COLOR_PP, linewidth=2.5, markersize=8,
             label="FJ Pure (actual)", zorder=5)
    ax2.plot(numa_count, pp_ideal, "--", color=COLOR_IDEAL, linewidth=1.5,
             label="Ideal linear", alpha=0.7)

    for i, (n, v) in enumerate(zip(numa_count, pp_vals)):
        ax2.annotate(f"{v:.1f}", (n, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color=COLOR_PP)

    ax2.set_xlabel("NUMA Nodes", fontsize=11)
    ax2.set_ylabel("Prefill Throughput (tokens/s)", fontsize=11)
    ax2.set_title("(b) Prefill (PP) Strong Scaling", fontsize=12, fontweight="bold")
    ax2.set_xticks(numa_count)
    ax2.legend(fontsize=9)
    ax2.grid(linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle("Strong Scaling: Fork-Join Pure (non-TP)\n"
                 "Qwen3-4B F16 | 24 threads/NUMA | Intel Xeon Platinum 8160",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Figure 2: Scaling Efficiency
# ============================================================================
def plot_scaling_efficiency(data, output_path, model="qwen4b"):
    """E(N) = T(N) / (N * T(1)) for decode and prefill."""
    numas = ["1n", "4n", "8n"]
    numa_count = [1, 4, 8]

    tg_vals = [get_val(data, n, "speed_tg", model=model) for n in numas]
    pp_vals = [get_val(data, n, "speed_pp", model=model) for n in numas]

    tg_eff = [tg_vals[i] / (numa_count[i] * tg_vals[0]) if tg_vals[0] > 0 else 0
              for i in range(len(numas))]
    pp_eff = [pp_vals[i] / (numa_count[i] * pp_vals[0]) if pp_vals[0] > 0 else 0
              for i in range(len(numas))]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(numa_count, tg_eff, "o-", color=COLOR_TG, linewidth=2.5, markersize=8,
            label="Decode (TG)")
    ax.plot(numa_count, pp_eff, "s-", color=COLOR_PP, linewidth=2.5, markersize=8,
            label="Prefill (PP)")
    ax.axhline(1.0, color=COLOR_IDEAL, linestyle="--", linewidth=1.5, alpha=0.7, label="Ideal (100%)")

    for i, (n, e_tg, e_pp) in enumerate(zip(numa_count, tg_eff, pp_eff)):
        ax.annotate(f"{e_tg:.1%}", (n, e_tg), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, color=COLOR_TG, fontweight="bold")
        ax.annotate(f"{e_pp:.1%}", (n, e_pp), textcoords="offset points",
                    xytext=(8, -12), fontsize=9, color=COLOR_PP, fontweight="bold")

    ax.set_xlabel("NUMA Nodes", fontsize=11)
    ax.set_ylabel("Scaling Efficiency E(N)", fontsize=11)
    ax.set_title("Strong Scaling Efficiency: E(N) = T(N) / [N * T(1)]", fontsize=12, fontweight="bold")
    ax.set_xticks(numa_count)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Figure 3: Barrier Amplification
# ============================================================================
def plot_barrier_amplification(data, output_path, model="qwen4b"):
    """
    A(N) = B(N) / B(1)   — barrier amplification
    C(N) = compute(N) / compute(1)  — compute amplification
    If A(N) >> C(N), synchronization is the bottleneck, not compute.
    """
    numas = ["1n", "4n", "8n"]
    numa_count = [1, 4, 8]

    barrier_vals = [get_val(data, n, "avg_barrier_us", model=model) for n in numas]
    compute_vals = [get_val(data, n, "avg_compute_us", model=model) for n in numas]
    graph_vals = [get_val(data, n, "avg_graph_us", model=model) for n in numas]

    if barrier_vals[0] > 0 and compute_vals[0] > 0:
        barrier_amp = [b / barrier_vals[0] for b in barrier_vals]
        compute_amp = [c / compute_vals[0] for c in compute_vals]
    else:
        print("WARNING: No barrier/compute data for 1N, skipping barrier amplification plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel (a): Amplification factors ---
    ax1.plot(numa_count, barrier_amp, "o-", color=COLOR_BARRIER, linewidth=2.5, markersize=8,
             label="Barrier Amplification A(N)")
    ax1.plot(numa_count, compute_amp, "s-", color=COLOR_COMPUTE, linewidth=2.5, markersize=8,
             label="Compute Amplification C(N)")
    ax1.axhline(1.0, color=COLOR_IDEAL, linestyle="--", linewidth=1.5, alpha=0.7, label="Baseline (1N)")

    for i, (n, ba, ca) in enumerate(zip(numa_count, barrier_amp, compute_amp)):
        ax1.annotate(f"{ba:.1f}x", (n, ba), textcoords="offset points",
                     xytext=(8, 5), fontsize=9, color=COLOR_BARRIER, fontweight="bold")
        ax1.annotate(f"{ca:.1f}x", (n, ca), textcoords="offset points",
                     xytext=(8, -12), fontsize=9, color=COLOR_COMPUTE, fontweight="bold")

    ax1.set_xlabel("NUMA Nodes", fontsize=11)
    ax1.set_ylabel("Amplification Factor (vs 1N)", fontsize=11)
    ax1.set_title("(a) Barrier vs Compute Amplification", fontsize=12, fontweight="bold")
    ax1.set_xticks(numa_count)
    ax1.legend(fontsize=9)
    ax1.grid(linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    # --- Panel (b): Stacked time breakdown ---
    barrier_pcts = [get_val(data, n, "barrier_pct", model=model) for n in numas]
    compute_pcts = [get_val(data, n, "compute_pct", model=model) for n in numas]
    idle_pcts = [100.0 - b - c for b, c in zip(barrier_pcts, compute_pcts)]

    x = np.arange(len(numas))
    width = 0.5

    ax2.bar(x, compute_pcts, width, color=COLOR_COMPUTE, alpha=0.85, label="Compute")
    ax2.bar(x, barrier_pcts, width, bottom=compute_pcts, color=COLOR_BARRIER, alpha=0.85, label="Barrier")
    ax2.bar(x, idle_pcts, width, bottom=[c + b for c, b in zip(compute_pcts, barrier_pcts)],
            color=COLOR_IDLE, alpha=0.85, label="Idle/Other")

    for i, (c, b) in enumerate(zip(compute_pcts, barrier_pcts)):
        ax2.text(i, c / 2, f"{c:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        ax2.text(i, c + b / 2, f"{b:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax2.set_xlabel("NUMA Configuration", fontsize=11)
    ax2.set_ylabel("Time Breakdown (%)", fontsize=11)
    ax2.set_title("(b) Compute / Barrier / Idle Breakdown", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n}N ({n * 24}t)" for n in numa_count])
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)

    fig.suptitle("Barrier Amplification Analysis (Decode, batch=1)\n"
                 "Qwen3-4B F16 | Fork-Join Pure | Intel Xeon 8160",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Figure 4: Thread Load Imbalance (Violin/Box plot)
# ============================================================================
def plot_thread_imbalance(thread_data, output_path, model="qwen4b"):
    """Per-thread compute time distribution, colored by NUMA domain."""
    numas = ["1n", "4n", "8n"]
    numa_threads = {"1n": 24, "4n": 96, "8n": 192}  # NUMA scaling: 24 threads per node
    threads_per_node = 24

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, numa in enumerate(numas):
        ax = axes[idx]
        expected_threads = numa_threads[numa]
        rows = [r for r in thread_data
                if r.get("numa") == numa and r.get("stage") == "decode"
                and int(r.get("threads", 0)) == expected_threads
                and (r.get("model") == model or r.get("model") in ["qwen3_4b", "qwen4b"])]

        if not rows:
            ax.set_title(f"{numa.upper()} — No data", fontsize=11)
            continue

        # Group by thread_id, compute mean
        thread_vals = {}
        thread_barrier = {}
        for r in rows:
            tid = int(r["thread_id"])
            c = float(r["compute_us"])
            b = float(r.get("barrier_us", 0))
            if tid not in thread_vals:
                thread_vals[tid] = []
                thread_barrier[tid] = []
            thread_vals[tid].append(c)
            thread_barrier[tid].append(b)

        nth = max(thread_vals.keys()) + 1
        means = np.array([np.mean(thread_vals.get(t, [0])) for t in range(nth)])
        barrier_means = np.array([np.mean(thread_barrier.get(t, [0])) for t in range(nth)])

        # Normalize to ms
        means_ms = means / 1000.0
        barrier_ms = barrier_means / 1000.0

        n_domains = int(numa[0])
        x = np.arange(nth)

        # Color by NUMA domain
        colors = [NUMA_COLORS[min(t // threads_per_node, n_domains - 1)] for t in range(nth)]

        # Stacked bar: compute + barrier
        bars_compute = ax.bar(x, means_ms, color=colors, alpha=0.8, width=0.8)
        bars_barrier = ax.bar(x, barrier_ms, bottom=means_ms, color="gray", alpha=0.4, width=0.8)

        # Stats annotation
        cv = np.std(means) / np.mean(means) * 100 if np.mean(means) > 0 else 0
        ratio = np.max(means) / np.min(means) if np.min(means) > 0 else 0
        avg_barrier_pct = np.mean(barrier_means) / (np.mean(means) + np.mean(barrier_means)) * 100 if (np.mean(means) + np.mean(barrier_means)) > 0 else 0

        ax.set_title(f"{numa.upper()} ({nth}t) | CoV={cv:.1f}% | Max/Min={ratio:.2f}x | Barrier={avg_barrier_pct:.0f}%",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Thread ID", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Time per graph call (ms)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        # Add mean line
        ax.axhline(np.mean(means_ms), color="black", linestyle="--", linewidth=1, alpha=0.5)

        # Legend for NUMA domains
        if n_domains > 1:
            patches = [mpatches.Patch(color=NUMA_COLORS[d], alpha=0.8, label=f"NUMA {d}")
                       for d in range(n_domains)]
            patches.append(mpatches.Patch(color="gray", alpha=0.4, label="Barrier"))
            ax.legend(handles=patches, fontsize=7, loc="upper right", ncol=2)

    fig.suptitle("Per-Thread Load Imbalance (Decode, batch=1)\n"
                 "Compute (colored) + Barrier (gray) | Fork-Join Pure",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Figure 5: Thread Sweep (8N fixed)
# ============================================================================
def plot_thread_sweep(data, output_path, model="qwen4b"):
    """
    Fixed 8N, vary thread count: 24/48/96/192.
    Three subplots: TG throughput, barrier%, idle%.
    """
    thread_counts = [24, 48, 96, 192]

    # Collect data for 8N thread sweep
    tg_vals = []
    barrier_pcts = []
    compute_pcts = []

    for tc in thread_counts:
        rows = [r for r in data
                if r.get("numa") == "8n" and r.get("stage") == "decode"
                and (r.get("model") == model or r.get("model") in ["qwen3_4b", "qwen4b"])
                and int(r.get("threads", 0)) == tc]
        if rows:
            tg_vals.append(float(rows[0].get("speed_tg", 0)))
            barrier_pcts.append(float(rows[0].get("barrier_pct", 0)))
            compute_pcts.append(float(rows[0].get("compute_pct", 0)))
        else:
            tg_vals.append(0)
            barrier_pcts.append(0)
            compute_pcts.append(0)

    if all(v == 0 for v in tg_vals):
        print("WARNING: No thread sweep data found for 8N, skipping")
        return

    idle_pcts = [100 - b - c for b, c in zip(barrier_pcts, compute_pcts)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # TG throughput
    ax1.plot(thread_counts, tg_vals, "o-", color=COLOR_TG, linewidth=2.5, markersize=8)
    for tc, v in zip(thread_counts, tg_vals):
        if v > 0:
            ax1.annotate(f"{v:.2f}", (tc, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Thread Count", fontsize=11)
    ax1.set_ylabel("Decode Throughput (tokens/s)", fontsize=11)
    ax1.set_title("(a) TG Throughput", fontsize=12, fontweight="bold")
    ax1.set_xticks(thread_counts)
    ax1.grid(linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    # Barrier %
    ax2.plot(thread_counts, barrier_pcts, "o-", color=COLOR_BARRIER, linewidth=2.5, markersize=8)
    for tc, v in zip(thread_counts, barrier_pcts):
        if v > 0:
            ax2.annotate(f"{v:.1f}%", (tc, v), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Thread Count", fontsize=11)
    ax2.set_ylabel("Barrier Time (%)", fontsize=11)
    ax2.set_title("(b) Barrier Overhead", fontsize=12, fontweight="bold")
    ax2.set_xticks(thread_counts)
    ax2.grid(linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)

    # Stacked breakdown
    x = np.arange(len(thread_counts))
    width = 0.6
    ax3.bar(x, compute_pcts, width, color=COLOR_COMPUTE, alpha=0.85, label="Compute")
    ax3.bar(x, barrier_pcts, width, bottom=compute_pcts, color=COLOR_BARRIER, alpha=0.85, label="Barrier")
    ax3.bar(x, idle_pcts, width, bottom=[c + b for c, b in zip(compute_pcts, barrier_pcts)],
            color=COLOR_IDLE, alpha=0.85, label="Idle")
    ax3.set_xlabel("Thread Count", fontsize=11)
    ax3.set_ylabel("Time Breakdown (%)", fontsize=11)
    ax3.set_title("(c) Time Breakdown", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(tc) for tc in thread_counts])
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 105)

    fig.suptitle("Thread Sweep on 8 NUMA Nodes (Decode, batch=1)\n"
                 "Qwen3-4B F16 | Fork-Join Pure | Intel Xeon 8160",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Figure 6: Per-op Barrier Contribution
# ============================================================================
def plot_per_op_barrier(output_path):
    """
    Parse ggml_perf_nodes.log and show top ops by barrier contribution.
    Only generated if the per-op data file exists.
    """
    node_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggml_perf_nodes.log")
    if not os.path.exists(node_file):
        print("INFO: No per-op data (ggml_perf_nodes.log), skipping per-op figure")
        return

    # Parse per-op data
    from collections import defaultdict
    op_barrier = defaultdict(list)
    op_compute = defaultdict(list)

    with open(node_file) as f:
        for line in f:
            if not line.startswith("GGML_PERF_NODE|"):
                continue
            parts = line.strip().split("|")
            rec = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    rec[k] = v

            op = rec.get("op", "UNKNOWN")
            nth = int(rec.get("nth", 0))

            total_barrier = sum(float(rec.get(f"t{t}_b", 0)) for t in range(nth))
            total_compute = sum(float(rec.get(f"t{t}_c", 0)) for t in range(nth))

            op_barrier[op].append(total_barrier / max(nth, 1))
            op_compute[op].append(total_compute / max(nth, 1))

    if not op_barrier:
        print("INFO: Per-op data empty, skipping")
        return

    # Aggregate: sum across all nodes of same op type
    ops = sorted(op_barrier.keys(), key=lambda o: -sum(op_barrier[o]))[:8]

    fig, ax = plt.subplots(figsize=(10, 5))

    barrier_sums = [sum(op_barrier[o]) / 1000.0 for o in ops]  # ms
    compute_sums = [sum(op_compute[o]) / 1000.0 for o in ops]

    x = np.arange(len(ops))
    width = 0.35

    ax.bar(x - width / 2, compute_sums, width, color=COLOR_COMPUTE, alpha=0.85, label="Compute (ms)")
    ax.bar(x + width / 2, barrier_sums, width, color=COLOR_BARRIER, alpha=0.85, label="Barrier (ms)")

    ax.set_xlabel("Op Type", fontsize=11)
    ax.set_ylabel("Total Time (ms)", fontsize=11)
    ax.set_title("Per-Op Barrier vs Compute Contribution (avg per graph call)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate scaling experiment figures")
    parser.add_argument("--summary", default="scaling_summary.csv", help="Summary CSV")
    parser.add_argument("--thread-data", default="scaling_per_thread.csv", help="Per-thread CSV")
    parser.add_argument("--output-dir", default=".", help="Output directory for figures")
    parser.add_argument("--model", default="qwen4b", help="Model name filter")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, args.summary)
    thread_path = os.path.join(base_dir, args.thread_data)
    out_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    summary_data = load_csv(summary_path) if os.path.exists(summary_path) else []
    thread_data = load_csv(thread_path) if os.path.exists(thread_path) else []

    if not summary_data:
        print(f"ERROR: No summary data found at {summary_path}")
        print("Run scaling_parse.py first.")
        sys.exit(1)

    print(f"Loaded {len(summary_data)} summary rows, {len(thread_data)} per-thread rows")

    # Generate figures
    plot_strong_scaling(summary_data, os.path.join(out_dir, "fig_strong_scaling.png"), model=args.model)
    plot_scaling_efficiency(summary_data, os.path.join(out_dir, "fig_scaling_efficiency.png"), model=args.model)
    plot_barrier_amplification(summary_data, os.path.join(out_dir, "fig_barrier_amplification.png"), model=args.model)

    if thread_data:
        plot_thread_imbalance(thread_data, os.path.join(out_dir, "fig_thread_imbalance.png"), model=args.model)
    else:
        print("INFO: No per-thread data, skipping imbalance figure")

    plot_thread_sweep(summary_data, os.path.join(out_dir, "fig_thread_sweep.png"), model=args.model)
    plot_per_op_barrier(os.path.join(out_dir, "fig_per_op_barrier.png"))

    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
