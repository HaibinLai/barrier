#!/usr/bin/env python3
"""
plot_exp_m1.py — Exp-M1: Per-thread kernel execution time distribution.

Key insight: We want to show **within-graph-call** thread variability,
not across-graph-call variability. For each graph call, all threads execute
the same set of kernels, but finish at different times due to NUMA effects.

Generates:
  1. Violin plot: per-thread mean compute (averaged across graph calls)
     showing inter-thread spread within the same workload.
  2. Per-graph-call spread analysis: how much does the slowest thread
     lag behind the fastest in each graph call.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(__file__) or "."
EXP_DIR = os.path.join(BASE_DIR, "exp_m1")

# NUMA node colors
NUMA_COLORS = ["#4285F4", "#EA4335", "#34A853", "#FBBC04",
               "#9C27B0", "#FF6D00", "#00BCD4", "#795548"]


def parse_perf_threads(filepath):
    """Parse GGML_PERF_THREADS lines. Returns list of dicts {tid: compute_us}."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("GGML_PERF_THREADS|"):
                continue
            parts = line.split("|")
            rec = {}
            nth = 0
            for p in parts[1:]:
                k, v = p.split("=")
                if k == "nth":
                    nth = int(v)
                else:
                    tid = int(k[1:])
                    rec[tid] = float(v)
            rec["_nth"] = nth
            records.append(rec)
    return records


def build_thread_matrix(records, skip_warmup=1, take_last_n=64):
    """Build matrix (n_samples, nth) of per-thread compute_us."""
    if not records:
        return 0, np.array([])
    recs = records[skip_warmup:]
    if len(recs) > take_last_n:
        recs = recs[-take_last_n:]
    nth = recs[0]["_nth"]
    matrix = np.zeros((len(recs), nth))
    for i, rec in enumerate(recs):
        for t in range(nth):
            matrix[i, t] = rec.get(t, 0)
    return nth, matrix


def get_numa_mapping(nth, n_numa_nodes, threads_per_node=24):
    mapping = []
    for t in range(nth):
        node = min(t // threads_per_node, n_numa_nodes - 1)
        mapping.append(node)
    return mapping


def plot_main_figure(matrix_1n, nth_1n, matrix_4n, nth_4n, output_path):
    """
    Main Exp-M1 figure: two subplots showing per-thread average compute time.

    For each thread, we average its compute time across all 64 TG graph calls.
    This shows the *systematic* difference between threads (NUMA effect),
    removing the per-graph-call variability.
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 9),
                              gridspec_kw={"height_ratios": [1, 2.8]})

    # ===== 1NUMA (24 threads) =====
    ax = axes[0]

    # Per-thread mean across graph calls (ms)
    thread_means_1n = np.mean(matrix_1n, axis=0) / 1000.0
    thread_stds_1n = np.std(matrix_1n, axis=0) / 1000.0

    x_1n = np.arange(nth_1n)
    bars = ax.bar(x_1n, thread_means_1n, color=NUMA_COLORS[1], alpha=0.8,
                  yerr=thread_stds_1n, capsize=2, error_kw={"linewidth": 0.8, "alpha": 0.5})

    # Compute inter-thread CV (of the means)
    cv_inter_1n = np.std(thread_means_1n) / np.mean(thread_means_1n) * 100
    spread_1n = (np.max(thread_means_1n) - np.min(thread_means_1n)) / np.mean(thread_means_1n) * 100

    ax.set_title(f"1NUMA · 24 threads (Node 1)  —  "
                 f"Inter-thread spread: {spread_1n:.1f}%  |  CV: {cv_inter_1n:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean compute\ntime (ms)", fontsize=10)
    ax.set_xticks(x_1n)
    ax.set_xticklabels([str(i) for i in x_1n], fontsize=7)
    ax.set_xlabel("Thread ID", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(np.mean(thread_means_1n), color="black", linestyle="--",
               linewidth=0.8, alpha=0.5)

    legend_1n = [mpatches.Patch(facecolor=NUMA_COLORS[1], alpha=0.8, label="Node 1")]
    ax.legend(handles=legend_1n, fontsize=9, loc="upper right")

    # ===== 4NUMA (96 threads) =====
    ax = axes[1]
    numa_map = get_numa_mapping(nth_4n, 4)

    thread_means_4n = np.mean(matrix_4n, axis=0) / 1000.0
    thread_stds_4n = np.std(matrix_4n, axis=0) / 1000.0

    x_4n = np.arange(nth_4n)
    colors = [NUMA_COLORS[numa_map[t]] for t in range(nth_4n)]
    bars = ax.bar(x_4n, thread_means_4n, color=colors, alpha=0.8,
                  yerr=thread_stds_4n, capsize=1, error_kw={"linewidth": 0.6, "alpha": 0.4})

    cv_inter_4n = np.std(thread_means_4n) / np.mean(thread_means_4n) * 100
    spread_4n = (np.max(thread_means_4n) - np.min(thread_means_4n)) / np.mean(thread_means_4n) * 100

    ax.set_title(f"4NUMA · 96 threads (Nodes 0–3)  —  "
                 f"Inter-thread spread: {spread_4n:.1f}%  |  CV: {cv_inter_4n:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean compute time (ms)", fontsize=10)
    ax.set_xlabel("Thread ID", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.axhline(np.mean(thread_means_4n), color="black", linestyle="--",
               linewidth=0.8, alpha=0.5)

    ax.set_xticks(range(0, nth_4n, 4))
    ax.set_xticklabels([str(i) for i in range(0, nth_4n, 4)], fontsize=7)

    # NUMA boundary lines + node labels
    for boundary in [24, 48, 72]:
        if boundary < nth_4n:
            ax.axvline(boundary - 0.5, color="gray", linestyle=":", alpha=0.6, linewidth=1.2)

    # Per-node mean annotation at top
    for n in range(4):
        threads = [t for t in range(nth_4n) if numa_map[t] == n]
        node_mean = np.mean(thread_means_4n[threads])
        mid = threads[len(threads)//2]
        ymax = ax.get_ylim()[1]
        ax.text(mid, ymax * 0.98, f"Node {n}\nmean={node_mean:.1f}ms",
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=NUMA_COLORS[n],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    legend_4n = [mpatches.Patch(facecolor=NUMA_COLORS[n], alpha=0.8, label=f"Node {n}")
                 for n in range(4)]
    ax.legend(handles=legend_4n, fontsize=9, loc="lower right", ncol=4)

    fig.suptitle(
        "Exp-M1: Per-Thread Compute Time (TG Decode) — LLaMA3-8B, batch=1\n"
        "Bar = mean across 64 graph calls · Error bar = std · Fork-Join · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_per_call_spread(matrix_1n, nth_1n, matrix_4n, nth_4n, output_path):
    """
    Chart 2: Per-graph-call thread spread.
    For each graph call, show (max_thread - min_thread) / min_thread as percentage.
    This directly measures how much idle time the barrier creates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (matrix, nth, label, color) in enumerate([
        (matrix_1n, nth_1n, "1NUMA (24t)", NUMA_COLORS[1]),
        (matrix_4n, nth_4n, "4NUMA (96t)", NUMA_COLORS[0]),
    ]):
        ax = axes[idx]
        n_calls = matrix.shape[0]

        # For each graph call: spread = (max - min) / min * 100
        maxs = np.max(matrix, axis=1)
        mins = np.min(matrix, axis=1)
        means = np.mean(matrix, axis=1)
        spread_pct = (maxs - mins) / mins * 100

        x = np.arange(n_calls)
        ax.bar(x, spread_pct, color=color, alpha=0.7)

        avg_spread = np.mean(spread_pct)
        ax.axhline(avg_spread, color="black", linestyle="--", linewidth=1.2)
        ax.text(n_calls * 0.98, avg_spread * 1.05, f"avg = {avg_spread:.1f}%",
                ha="right", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(f"{label}  —  Avg spread: {avg_spread:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Thread spread\n(max−min)/min (%)", fontsize=10)
        ax.set_xlabel("Graph call index (TG decode)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Per-Graph-Call Thread Execution Spread — LLaMA3-8B (batch=1)\n"
        "Higher = more idle time in barrier · Fork-Join · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def plot_violin_within_call(matrix_1n, nth_1n, matrix_4n, nth_4n, output_path):
    """
    Chart 3: Violin plot of per-thread compute times within each graph call.
    For each thread, normalize by the graph-call mean, showing relative speed.
    This removes the PP vs TG difference and focuses on inter-thread variability.
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 9),
                              gridspec_kw={"height_ratios": [1, 2.8]})

    for idx, (matrix, nth, n_nodes, title_prefix) in enumerate([
        (matrix_1n, nth_1n, 1, "1NUMA · 24 threads (Node 1)"),
        (matrix_4n, nth_4n, 4, "4NUMA · 96 threads (Nodes 0–3)"),
    ]):
        ax = axes[idx]
        numa_map = get_numa_mapping(nth, n_nodes)

        # Normalize: for each graph call, divide each thread's time by the call's mean
        call_means = np.mean(matrix, axis=1, keepdims=True)  # (n_calls, 1)
        normalized = matrix / call_means  # ratio: >1 = slower than average, <1 = faster

        # Per-thread violin of normalized values
        data = [normalized[:, t] for t in range(nth)]

        parts = ax.violinplot(data, positions=range(nth), showmeans=True,
                              showmedians=False, showextrema=False, widths=0.8)

        for i, pc in enumerate(parts["bodies"]):
            node = numa_map[i] if n_nodes > 1 else (1 if n_nodes == 1 else 0)
            pc.set_facecolor(NUMA_COLORS[node])
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(0.5)

        # Reference line at 1.0
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # Spread metric
        thread_mean_ratios = np.mean(normalized, axis=0)
        spread = (np.max(thread_mean_ratios) - np.min(thread_mean_ratios)) * 100

        ax.set_title(f"{title_prefix}  —  Systematic inter-thread spread: {spread:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Normalized compute\n(ratio to call mean)", fontsize=10)

        if nth <= 24:
            ax.set_xticks(range(nth))
            ax.set_xticklabels([str(i) for i in range(nth)], fontsize=7)
        else:
            ax.set_xticks(range(0, nth, 4))
            ax.set_xticklabels([str(i) for i in range(0, nth, 4)], fontsize=7)
        ax.set_xlabel("Thread ID", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        # NUMA boundaries
        if n_nodes > 1:
            for boundary in range(24, nth, 24):
                ax.axvline(boundary - 0.5, color="gray", linestyle=":", alpha=0.6, linewidth=1.2)
            # Node labels
            for n in range(n_nodes):
                threads = [t for t in range(nth) if numa_map[t] == n]
                mid = threads[len(threads)//2]
                legend_items = [mpatches.Patch(facecolor=NUMA_COLORS[nn], alpha=0.7,
                                label=f"Node {nn}") for nn in range(n_nodes)]
            ax.legend(handles=legend_items, fontsize=8, loc="upper right", ncol=n_nodes)
        else:
            legend_items = [mpatches.Patch(facecolor=NUMA_COLORS[1], alpha=0.7, label="Node 1")]
            ax.legend(handles=legend_items, fontsize=9, loc="upper right")

    fig.suptitle(
        "Exp-M1: Within-Call Normalized Thread Compute — LLaMA3-8B (batch=1, TG decode)\n"
        "Each thread's compute / graph-call mean · >1 = slower than average · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    file_1n = os.path.join(EXP_DIR, "1numa_llama8b_pl1.log")
    file_4n = os.path.join(EXP_DIR, "4numa_llama8b_pl1.log")

    print("Parsing 1NUMA data...")
    records_1n = parse_perf_threads(file_1n)
    nth_1n, matrix_1n = build_thread_matrix(records_1n, skip_warmup=1, take_last_n=64)
    print(f"  1NUMA: {nth_1n} threads, {matrix_1n.shape[0]} samples")

    print("Parsing 4NUMA data...")
    records_4n = parse_perf_threads(file_4n)
    nth_4n, matrix_4n = build_thread_matrix(records_4n, skip_warmup=1, take_last_n=64)
    print(f"  4NUMA: {nth_4n} threads, {matrix_4n.shape[0]} samples")

    # Chart 1: Per-thread mean compute bar chart
    plot_main_figure(matrix_1n, nth_1n, matrix_4n, nth_4n,
                     os.path.join(BASE_DIR, "exp_m1_thread_means.png"))

    # Chart 2: Per-graph-call spread
    plot_per_call_spread(matrix_1n, nth_1n, matrix_4n, nth_4n,
                         os.path.join(BASE_DIR, "exp_m1_spread.png"))

    # Chart 3: Normalized violin (within-call variability)
    plot_violin_within_call(matrix_1n, nth_1n, matrix_4n, nth_4n,
                            os.path.join(BASE_DIR, "exp_m1_violin_normalized.png"))


if __name__ == "__main__":
    main()
