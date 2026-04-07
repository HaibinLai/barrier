#!/usr/bin/env python3
"""
plot_14b_latency.py — Latency view of 14B experiment results.

Figures:
  1. TTFT (Time to First Token) across NUMA configs
  2. TPOT (Time Per Output Token) across NUMA configs
  3. E2E latency across NUMA configs
  4. Latency breakdown: stacked bar (TTFT + decode time)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))

def load_last(fname):
    path = os.path.join(BASE, fname)
    try:
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip().startswith("{")]
        return json.loads(lines[-1]) if lines else None
    except FileNotFoundError:
        return None

def load_all(fname):
    path = os.path.join(BASE, fname)
    results = []
    try:
        with open(path) as f:
            for l in f:
                l = l.strip()
                if l.startswith("{"):
                    results.append(json.loads(l))
    except FileNotFoundError:
        pass
    return results

# ── Collect decode data ──
configs = [
    ("1N FJ-Pure",        "qwen25_14b_1n_fj_decode_r1.jsonl",       "#d62728"),
    ("1N TaskInfer",      "qwen25_14b_1n_task_decode_r1.jsonl",      "#9467bd"),
    ("4N FJ-Pure",        "qwen25_14b_4n_fj_decode_r1.jsonl",        "#d62728"),
    ("4N FJ+TP4\n(no RS)","qwen25_14b_4n_fjtp_decode_r1.jsonl",      "#ff7f0e"),
    ("4N FJ+TP4\n+RS",    "qwen25_14b_4n_fjtp_rs_decode_r1.jsonl",   "#1f77b4"),
    ("4N TaskInfer\n+RS",  "qwen25_14b_4n_task_rs_decode_r1.jsonl",  "#2ca02c"),
    ("8N FJ-Pure",        "qwen25_14b_8n_fj_decode_r1.jsonl",        "#d62728"),
    ("8N FJ+TP8\n(no RS)","qwen25_14b_8n_fjtp_decode_r1.jsonl",      "#ff7f0e"),
    ("8N FJ+TP8\n+RS",    "qwen25_14b_8n_fjtp_rs_decode_r1.jsonl",   "#1f77b4"),
    ("8N TaskInfer\n+RS",  "qwen25_14b_8n_task_rs_decode_r1.jsonl",  "#2ca02c"),
]

data = []
for label, fname, color in configs:
    d = load_last(fname)
    if d:
        tpot = d["t_tg"] / d["tg"] if d["tg"] > 0 else 0
        data.append({
            "label": label, "color": color,
            "ttft": d["t_pp"], "tpot": tpot,
            "t_tg": d["t_tg"], "e2e": d["t"],
            "pp": d["pp"], "tg": d["tg"],
            "speed_pp": d["speed_pp"], "speed_tg": d["speed_tg"],
        })

# ── Figure 1: TTFT ──
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(data))
ttfts = [d["ttft"] for d in data]
colors = [d["color"] for d in data]
bars = ax.bar(x, ttfts, color=colors, alpha=0.85, width=0.7, edgecolor="white", linewidth=0.5)
for bar, v in zip(bars, ttfts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([d["label"] for d in data], fontsize=8)
ax.set_ylabel("TTFT — Time to First Token (seconds)", fontsize=11)
ax.set_title("Qwen2.5-14B (F16) — Time to First Token (TTFT)\n"
             "Lower is better · PP=64 · Intel Xeon Platinum 8160",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)

# Add NUMA separators
for xpos in [1.5, 5.5]:
    ax.axvline(xpos, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax.text(0.5, ax.get_ylim()[1]*0.95, "1 NUMA", ha="center", fontsize=10, color="gray", style="italic")
ax.text(3.5, ax.get_ylim()[1]*0.95, "4 NUMA", ha="center", fontsize=10, color="gray", style="italic")
ax.text(7.5, ax.get_ylim()[1]*0.95, "8 NUMA", ha="center", fontsize=10, color="gray", style="italic")

fig.tight_layout()
fig.savefig(os.path.join(BASE, "fig_14b_ttft.png"), dpi=180, bbox_inches="tight")
print("Saved fig_14b_ttft.png")

# ── Figure 2: TPOT ──
fig, ax = plt.subplots(figsize=(14, 5))
tpots = [d["tpot"] for d in data]
bars = ax.bar(x, tpots, color=colors, alpha=0.85, width=0.7, edgecolor="white", linewidth=0.5)
for bar, v in zip(bars, tpots):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{v:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([d["label"] for d in data], fontsize=8)
ax.set_ylabel("TPOT — Time Per Output Token (seconds)", fontsize=11)

# Note about different TG values
ax.set_title("Qwen2.5-14B (F16) — Time Per Output Token (TPOT)\n"
             "Lower is better · 1N: TG=64, 4N/8N: TG=8 · Intel Xeon Platinum 8160",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.set_axisbelow(True)
for xpos in [1.5, 5.5]:
    ax.axvline(xpos, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax.text(0.5, ax.get_ylim()[1]*0.95, "1 NUMA", ha="center", fontsize=10, color="gray", style="italic")
ax.text(3.5, ax.get_ylim()[1]*0.95, "4 NUMA", ha="center", fontsize=10, color="gray", style="italic")
ax.text(7.5, ax.get_ylim()[1]*0.95, "8 NUMA", ha="center", fontsize=10, color="gray", style="italic")

fig.tight_layout()
fig.savefig(os.path.join(BASE, "fig_14b_tpot.png"), dpi=180, bbox_inches="tight")
print("Saved fig_14b_tpot.png")

# ── Figure 3: Latency breakdown (stacked: TTFT + decode) ──
# Only compare same-TG configs (4N/8N, TG=8)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for idx, (numa_label, start_idx, end_idx) in enumerate([("4 NUMA", 2, 6), ("8 NUMA", 6, 10)]):
    ax = axes[idx]
    subset = data[start_idx:end_idx]
    n = len(subset)
    xx = np.arange(n)

    ttft_vals = [d["ttft"] for d in subset]
    decode_vals = [d["t_tg"] for d in subset]
    colors_sub = [d["color"] for d in subset]

    bars1 = ax.bar(xx, ttft_vals, width=0.6, color=colors_sub, alpha=0.9,
                   label="TTFT (prefill)", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(xx, decode_vals, width=0.6, bottom=ttft_vals,
                   color=colors_sub, alpha=0.45, hatch="///",
                   label="Decode time", edgecolor="white", linewidth=0.5)

    for i, (t1, t2, d_) in enumerate(zip(ttft_vals, decode_vals, subset)):
        total = t1 + t2
        # TTFT label
        if t1 > 5:
            ax.text(i, t1/2, f"TTFT\n{t1:.1f}s", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        # Total on top
        ax.text(i, total + 2, f"E2E\n{d_['e2e']:.1f}s", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(xx)
    ax.set_xticklabels([d["label"] for d in subset], fontsize=8)
    tp_n = "4" if idx == 0 else "8"
    n_t = "96" if idx == 0 else "192"
    ax.set_title(f"{numa_label} ({n_t} threads)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (seconds)" if idx == 0 else "", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.9, label="TTFT (prefill)"),
        Patch(facecolor="gray", alpha=0.45, hatch="///", label="Decode (TG=8)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

fig.suptitle("Qwen2.5-14B (F16) — Latency Breakdown\n"
             "PP=64 TG=8 · Stacked: TTFT + Decode Time · Intel Xeon Platinum 8160",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(BASE, "fig_14b_latency_breakdown.png"), dpi=180, bbox_inches="tight")
print("Saved fig_14b_latency_breakdown.png")

# ── Figure 4: RS speedup on latency metrics ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = [
    ("TTFT Reduction", "ttft", "s"),
    ("TPOT Reduction", "tpot", "s"),
    ("E2E Reduction", "e2e", "s"),
]

for ax, (title, key, unit) in zip(axes, metrics):
    comparisons = []

    for numa, base_sys, rs_systems in [
        ("4N", "4N FJ-Pure", [("4N FJ+TP4\n+RS", "#1f77b4"), ("4N TaskInfer\n+RS", "#2ca02c")]),
        ("8N", "8N FJ-Pure", [("8N FJ+TP8\n+RS", "#1f77b4"), ("8N TaskInfer\n+RS", "#2ca02c")]),
    ]:
        base_val = next((d[key] for d in data if d["label"] == base_sys), 0)
        for rs_label, rs_color in rs_systems:
            rs_val = next((d[key] for d in data if d["label"] == rs_label), 0)
            if base_val > 0:
                reduction = base_val / rs_val
                short_label = rs_label.replace("\n", " ")
                comparisons.append((short_label, reduction, rs_color))

    xx = np.arange(len(comparisons))
    labels = [c[0] for c in comparisons]
    vals = [c[1] for c in comparisons]
    cols = [c[2] for c in comparisons]

    bars = ax.bar(xx, vals, color=cols, alpha=0.85, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{v:.1f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(xx)
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Reduction Factor (x)" if ax == axes[0] else "", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle("Qwen2.5-14B — Latency Reduction with row-shard-cache\n"
             "vs FJ-Pure baseline · Higher = faster · Intel Xeon Platinum 8160",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(BASE, "fig_14b_latency_reduction.png"), dpi=180, bbox_inches="tight")
print("Saved fig_14b_latency_reduction.png")

# ── Summary table ──
print("\n" + "=" * 80)
print("LATENCY SUMMARY TABLE (for paper)")
print("=" * 80)
print(f"\n{'Config':<22} {'TTFT(s)':<10} {'TPOT(s)':<10} {'E2E(s)':<10}")
print("-" * 55)
for d in data:
    label = d["label"].replace("\n", " ")
    print(f"{label:<22} {d['ttft']:<10.2f} {d['tpot']:<10.3f} {d['e2e']:<10.1f}")

print("\nNote: 1N uses TG=64, 4N/8N use TG=8")
print("      TPOT and E2E not directly comparable across NUMA configs")
print("      TTFT is comparable (all use PP=64)")
