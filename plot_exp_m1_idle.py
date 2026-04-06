#!/usr/bin/env python3
"""
plot_exp_m1_idle.py — Exp-M1: Per-thread compute + idle stacked bar.

Shows 1NUMA, 4NUMA, 8NUMA for two models.
idle_time = max_compute_across_threads - this_thread_compute
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(__file__) or "."
EXP_DIR = os.path.join(BASE_DIR, "exp_m1")

NUMA_COLORS = ["#4285F4", "#EA4335", "#34A853", "#FBBC04",
               "#9C27B0", "#FF6D00", "#00BCD4", "#795548"]
IDLE_COLOR = "#E0E0E0"


def parse_perf_threads(filepath):
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
                    rec[int(k[1:])] = float(v)
            rec["_nth"] = nth
            records.append(rec)
    return records


def build_thread_matrix(records, skip_warmup=1, take_last_n=64):
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


def get_numa_color(tid, n_nodes, tpn=24):
    node = min(tid // tpn, n_nodes - 1)
    return NUMA_COLORS[node], node


def plot_average_idle(matrix, nth, n_nodes, ax, title, show_ylabel=True):
    """Plot average across all TG graph calls: mean compute + mean idle per thread."""
    n_calls = matrix.shape[0]
    compute_ms = matrix / 1000.0

    call_max = np.max(compute_ms, axis=1, keepdims=True)
    idle_ms = call_max - compute_ms

    avg_compute = np.mean(compute_ms, axis=0)
    avg_idle = np.mean(idle_ms, axis=0)

    # Sort by avg_compute (ascending)
    sorted_idx = np.argsort(avg_compute)
    avg_compute_s = avg_compute[sorted_idx]
    avg_idle_s = avg_idle[sorted_idx]

    x = np.arange(nth)

    for i, orig_tid in enumerate(sorted_idx):
        color, node = get_numa_color(orig_tid, n_nodes)
        ax.bar(i, avg_compute_s[i], color=color, alpha=0.85, width=0.9)

    ax.bar(x, avg_idle_s, bottom=avg_compute_s, color=IDLE_COLOR, alpha=0.8, width=0.9,
           edgecolor="#BDBDBD", linewidth=0.3)

    total_height = avg_compute_s + avg_idle_s
    ax.axhline(np.max(total_height), color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    total_compute = np.sum(avg_compute)
    total_idle = np.sum(avg_idle)
    idle_pct = total_idle / (total_compute + total_idle) * 100

    ax.set_title(title, fontsize=11, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Time (ms)", fontsize=10)
    ax.set_xlabel("Threads (sorted by compute time)", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.text(0.97, 0.95, f"Idle: {idle_pct:.1f}%\nof total CPU time",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            fontweight="bold", color="#757575",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    if nth > 48:
        step = max(nth // 12, 1)
        ax.set_xticks(range(0, nth, step))
        ax.set_xticklabels([str(i) for i in range(0, nth, step)], fontsize=6)
    elif nth > 24:
        ax.set_xticks(range(0, nth, 4))
        ax.set_xticklabels([str(i) for i in range(0, nth, 4)], fontsize=7)
    else:
        ax.set_xticks(range(0, nth, 2))
        ax.set_xticklabels([str(i) for i in range(0, nth, 2)], fontsize=7)


def main():
    models = [
        ("Qwen3-4B", "qwen4b"),
        ("LLaMA3-8B", "llama8b"),
    ]

    numa_configs = [
        ("1NUMA (24t)", "1numa", 1),
        ("4NUMA (96t)", "4numa", 4),
        ("8NUMA (192t)", "8numa", 8),
    ]

    # 3 rows (NUMA configs) x 2 cols (models)
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))

    for col, (display_name, file_key) in enumerate(models):
        for row, (numa_label, numa_prefix, n_nodes) in enumerate(numa_configs):
            ax = axes[row][col]
            filepath = os.path.join(EXP_DIR, f"{numa_prefix}_{file_key}_pl1.log")
            records = parse_perf_threads(filepath)
            nth, matrix = build_thread_matrix(records, skip_warmup=1, take_last_n=64)

            plot_average_idle(matrix, nth, n_nodes, ax,
                              f"{display_name} · {numa_label}",
                              show_ylabel=(col == 0))

    # Legend
    legend_items = [mpatches.Patch(facecolor=NUMA_COLORS[n], alpha=0.85, label=f"Node {n}")
                    for n in range(8)]
    legend_items.append(mpatches.Patch(facecolor=IDLE_COLOR, alpha=0.8,
                                        edgecolor="#BDBDBD", label="Idle (waiting at barrier)"))
    fig.legend(handles=legend_items, fontsize=9, loc="lower center",
               ncol=9, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Per-Thread Compute vs Idle Time (avg over 64 TG decode calls, batch=1)\n"
        "Colored = useful compute · Gray = idle waiting at barrier · Fork-Join · Intel Xeon 8160",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "exp_m1_idle_1v4v8.png"), dpi=180, bbox_inches="tight")
    print("Saved: exp_m1_idle_1v4v8.png")


if __name__ == "__main__":
    main()
