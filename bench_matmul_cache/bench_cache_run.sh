#!/bin/bash
# =================================================================
# bench_cache_run.sh — Build and run matmul kernel cache microbenchmark
#
# Measures single-core vec_dot_f16 throughput vs tile working-set size.
# Expected to show a "staircase" drop at L1d (32KB) and L2 (1MB).
#
# Output: bench_matmul_cache/results/bench_cache.csv
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$SCRIPT_DIR"
RESULT_DIR="$BENCH_DIR/results"
mkdir -p "$RESULT_DIR"

# ── Paths to ggml build (using exp_gran_forkjoin which has no llamafile) ──
GGML_ROOT="/home/haibin/tp/exp_gran_forkjoin/llama.cpp"
GGML_INC="$GGML_ROOT/ggml/include"
GGML_CPU_INC="$GGML_ROOT/ggml/src/ggml-cpu"
GGML_LIB="$GGML_ROOT/build/bin"

BENCH_SRC="$BENCH_DIR/bench_cache.c"
BENCH_BIN="$BENCH_DIR/bench_cache"
BENCH_CSV="$RESULT_DIR/bench_cache.csv"

NUMA="numactl --cpunodebind=0 --membind=0"

# ── Step 1: Build ──
echo "=== Building bench_cache ==="
gcc -O2 -march=native -o "$BENCH_BIN" "$BENCH_SRC" \
    -I"$GGML_INC" \
    -I"$GGML_CPU_INC" \
    -L"$GGML_LIB" \
    -lggml-cpu -lggml-base -lm \
    -Wl,-rpath,"$GGML_LIB"

echo "Built: $BENCH_BIN"

# ── Step 2: Run ──
echo "=== Running on NUMA node 0 (single core) ==="
echo "Output: $BENCH_CSV"
echo ""

# Pin to single core (core 0) for deterministic results
$NUMA taskset -c 0 "$BENCH_BIN" > "$BENCH_CSV"

echo ""
echo "=== Results saved to $BENCH_CSV ==="
echo ""
echo "To plot: python3 bench_matmul_cache/bench_cache_plot.py"
