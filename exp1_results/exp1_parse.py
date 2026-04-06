#!/usr/bin/env python3
"""
exp1_parse.py — Parse Experiment 1 raw JSONL results into summary CSV.

Reads all .jsonl files from exp1_results/raw/ and produces:
  - exp1_results/exp1_summary.csv: full data with median across repeats
  - Prints formatted table to stdout
"""

import os
import json
import csv
import sys
from collections import defaultdict
from statistics import median, stdev

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUT_CSV = os.path.join(os.path.dirname(__file__), "exp1_summary.csv")


def parse_jsonl(filepath):
    """Parse JSONL benchmark output file. Returns list of result dicts."""
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def parse_filename(filename):
    """
    Parse filename like: qwen3_4b_4n_fjtp_decode_r1.jsonl
    Returns: (model, numa, system, stage, repeat)
    """
    base = filename.replace(".jsonl", "")
    parts = base.split("_")

    # Find model name (first 2-3 parts until numa config)
    # Model names: qwen3_4b, llama3_8b, qwen25_14b
    for i, p in enumerate(parts):
        if p in ("1n", "4n", "8n"):
            model = "_".join(parts[:i])
            numa = p
            rest = parts[i+1:]
            break
    else:
        return None

    # rest = ["fj", "decode", "r1"] or ["fjtp", "prefill", "r2"] or ["task", "batch", "r3"]
    system = rest[0]  # fj, fjtp, task
    stage = rest[1]   # decode, prefill, batch
    repeat = int(rest[2][1:])  # r1 -> 1

    return model, numa, system, stage, repeat


def main():
    # Collect all data
    all_data = defaultdict(list)  # (model, numa, system, stage, pl) -> [(repeat, speed_pp, speed_tg)]

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".jsonl"):
            continue

        parsed = parse_filename(fname)
        if parsed is None:
            print(f"WARNING: cannot parse filename: {fname}", file=sys.stderr)
            continue

        model, numa, system, stage, repeat = parsed
        filepath = os.path.join(RAW_DIR, fname)
        results = parse_jsonl(filepath)

        for r in results:
            key = (model, numa, system, stage, r.get("pl", 1))
            all_data[key].append({
                "repeat": repeat,
                "speed_pp": r.get("speed_pp", 0),
                "speed_tg": r.get("speed_tg", 0),
                "speed_total": r.get("speed", 0),
                "t_pp": r.get("t_pp", 0),
                "t_tg": r.get("t_tg", 0),
                "pp": r.get("pp", 0),
                "tg": r.get("tg", 0),
            })

    # Compute medians
    rows = []
    for key, measurements in sorted(all_data.items()):
        model, numa, system, stage, pl = key

        speeds_pp = [m["speed_pp"] for m in measurements]
        speeds_tg = [m["speed_tg"] for m in measurements]
        speeds_total = [m["speed_total"] for m in measurements]

        row = {
            "model": model,
            "numa": numa,
            "system": system,
            "stage": stage,
            "batch": pl,
            "n_repeats": len(measurements),
            "pp_tokens": measurements[0]["pp"],
            "tg_tokens": measurements[0]["tg"],
            "speed_pp_median": round(median(speeds_pp), 2),
            "speed_tg_median": round(median(speeds_tg), 3),
            "speed_total_median": round(median(speeds_total), 2),
            "speed_pp_std": round(stdev(speeds_pp), 2) if len(speeds_pp) > 1 else 0,
            "speed_tg_std": round(stdev(speeds_tg), 3) if len(speeds_tg) > 1 else 0,
        }
        rows.append(row)

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Written {len(rows)} rows to {OUT_CSV}")

    # Print formatted summary table
    print("\n" + "="*100)
    print("EXPERIMENT 1 SUMMARY — Median across repeats")
    print("="*100)

    # Group by stage for cleaner output
    for stage in ["decode", "prefill", "batch"]:
        stage_rows = [r for r in rows if r["stage"] == stage]
        if not stage_rows:
            continue

        print(f"\n{'='*80}")
        print(f"  STAGE: {stage.upper()}")
        print(f"{'='*80}")
        print(f"{'Model':<15} {'NUMA':<6} {'System':<8} {'Batch':<6} "
              f"{'PP t/s':<10} {'TG t/s':<10} {'Total t/s':<10} {'N':<4}")
        print("-" * 80)

        for r in sorted(stage_rows, key=lambda x: (x["model"], x["numa"], x["batch"], x["system"])):
            print(f"{r['model']:<15} {r['numa']:<6} {r['system']:<8} {r['batch']:<6} "
                  f"{r['speed_pp_median']:<10} {r['speed_tg_median']:<10} "
                  f"{r['speed_total_median']:<10} {r['n_repeats']:<4}")


if __name__ == "__main__":
    main()
