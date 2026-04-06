#!/usr/bin/env python3
"""
plot_exp_m1_combined.py — Exp-M1: Combined multi-model per-thread variability figure.
Shows LLaMA3-8B and Qwen3-4B side by side for 1NUMA vs 4NUMA.

Key figure for Motivation Section 3.1.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(__file__) or "."
EXP_DIR = os.path.join(BASE_DIR, "exp_m1")

NUMA_COLORS = ["#4285F4", "#EA4335", "#34A853", "#FBBC04"]


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


def get_numa_mapping(nth, n_nodes, tpn=24):
    return [min(t // tpn, n_nodes - 1) for t in range(nth)]


def main():
    models = [
        ("Qwen3-4B", "qwen4b"),
        ("LLaMA3-8B", "llama8b"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    # Row 0: 1NUMA, Row 1: 4NUMA
    # Col 0: Qwen3-4B, Col 1: LLaMA3-8B

    for col, (display_name, file_key) in enumerate(models):
        for row, (numa_label, numa_prefix, n_nodes, node_color_base) in enumerate([
            ("1NUMA (24t)", "1numa", 1, 1),     # cpunodebind=1 → Node 1
            ("4NUMA (96t)", "4numa", 4, 0),
        ]):
            ax = axes[row][col]
            filepath = os.path.join(EXP_DIR, f"{numa_prefix}_{file_key}_pl1.log")
            records = parse_perf_threads(filepath)
            nth, matrix = build_thread_matrix(records, skip_warmup=1, take_last_n=64)

            numa_map = get_numa_mapping(nth, n_nodes)

            # Per-thread mean (ms)
            thread_means = np.mean(matrix, axis=0) / 1000.0
            thread_stds = np.std(matrix, axis=0) / 1000.0

            x = np.arange(nth)
            if n_nodes == 1:
                colors = [NUMA_COLORS[node_color_base]] * nth
            else:
                colors = [NUMA_COLORS[numa_map[t]] for t in range(nth)]

            ax.bar(x, thread_means, color=colors, alpha=0.8,
                   yerr=thread_stds, capsize=1 if nth > 30 else 2,
                   error_kw={"linewidth": 0.5 if nth > 30 else 0.8, "alpha": 0.4})

            # Inter-thread metrics
            cv = np.std(thread_means) / np.mean(thread_means) * 100
            spread = (np.max(thread_means) - np.min(thread_means)) / np.mean(thread_means) * 100

            # Per-graph-call spread
            maxs = np.max(matrix, axis=1)
            mins = np.min(matrix, axis=1)
            avg_call_spread = np.mean((maxs - mins) / mins * 100)

            title_parts = [f"{numa_label}"]
            title_parts.append(f"Thread spread: {spread:.1f}%  |  Per-call spread: {avg_call_spread:.1f}%")
            ax.set_title("\n".join(title_parts), fontsize=10, fontweight="bold")

            ax.set_ylabel("Mean compute (ms)", fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)
            ax.axhline(np.mean(thread_means), color="black", linestyle="--",
                       linewidth=0.8, alpha=0.4)

            if nth <= 24:
                ax.set_xticks(x)
                ax.set_xticklabels([str(i) for i in x], fontsize=6)
            else:
                ax.set_xticks(range(0, nth, 8))
                ax.set_xticklabels([str(i) for i in range(0, nth, 8)], fontsize=6)

            if row == 1:
                ax.set_xlabel("Thread ID", fontsize=9)

            # NUMA boundaries for 4NUMA
            if n_nodes > 1:
                for boundary in range(24, nth, 24):
                    ax.axvline(boundary - 0.5, color="gray", linestyle=":", alpha=0.6)

            # Column title (model name)
            if row == 0:
                ax.text(0.5, 1.25, display_name, transform=ax.transAxes,
                        fontsize=14, fontweight="bold", ha="center", va="bottom")

    # Legend
    legend_items = [mpatches.Patch(facecolor=NUMA_COLORS[n], alpha=0.8, label=f"Node {n}")
                    for n in range(4)]
    fig.legend(handles=legend_items, fontsize=10, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Exp-M1: Per-Thread Compute Time Distribution (TG Decode, batch=1)\n"
        "Fork-Join · PP64+TG64 · Intel Xeon 8160",
        fontsize=14, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "exp_m1_combined.png"), dpi=180, bbox_inches="tight")
    print("Saved: exp_m1_combined.png")

    # ===== Chart 2: Per-call spread comparison (key figure) =====
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))

    for col, (display_name, file_key) in enumerate(models):
        ax = axes2[col]

        spread_data = {}
        for numa_label, numa_prefix, n_nodes, color in [
            ("1NUMA (24t)", "1numa", 1, NUMA_COLORS[1]),
            ("4NUMA (96t)", "4numa", 4, NUMA_COLORS[0]),
        ]:
            filepath = os.path.join(EXP_DIR, f"{numa_prefix}_{file_key}_pl1.log")
            records = parse_perf_threads(filepath)
            nth, matrix = build_thread_matrix(records, skip_warmup=1, take_last_n=64)

            maxs = np.max(matrix, axis=1)
            mins = np.min(matrix, axis=1)
            spread_pct = (maxs - mins) / mins * 100
            spread_data[numa_label] = (spread_pct, color)

        n_calls = 64
        x = np.arange(n_calls)
        width = 0.4

        for i, (label, (data, color)) in enumerate(spread_data.items()):
            offset = -width/2 + i * width
            bars = ax.bar(x + offset, data[:n_calls], width, color=color, alpha=0.7, label=label)

        # Average lines
        for label, (data, color) in spread_data.items():
            avg = np.mean(data[:n_calls])
            ax.axhline(avg, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
            y_offset = 5 if "1NUMA" in label else -10
            ax.text(n_calls * 0.98, avg + y_offset, f"{label}: avg={avg:.0f}%",
                    ha="right", va="bottom" if "1NUMA" in label else "top",
                    fontsize=9, fontweight="bold", color=color)

        ax.set_title(display_name, fontsize=13, fontweight="bold")
        ax.set_ylabel("Thread spread\n(max−min)/min (%)", fontsize=10)
        ax.set_xlabel("Graph call index (TG decode)", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, loc="upper left")

    fig2.suptitle(
        "Per-Graph-Call Thread Execution Spread — batch=1, TG Decode\n"
        "Higher = more idle time at barrier · Fork-Join · Intel Xeon 8160",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig2.tight_layout()
    fig2.savefig(os.path.join(BASE_DIR, "exp_m1_spread_combined.png"), dpi=180, bbox_inches="tight")
    print("Saved: exp_m1_spread_combined.png")


if __name__ == "__main__":
    main()
