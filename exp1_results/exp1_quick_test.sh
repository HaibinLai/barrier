#!/bin/bash
# =================================================================
# exp1_quick_test.sh — Quick validation: 1 model, 4N, all 3 groups
# Verify everything works before full experiment
# Expected runtime: ~5 minutes
# =================================================================
set -euo pipefail

FJ_BIN=/home/haibin/tp/fork-join/build/bin/llama-batched-bench
TF_BIN=/home/haibin/tp/tp-llama/build/bin/llama-batched-bench

MODEL=$HOME/model/Qwen3-4B-Ins-2507-f16.gguf
OUTDIR=/home/haibin/tp/exp1_results/quick_test
mkdir -p "$OUTDIR"

NUMA="numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3"
THREADS=96
TP=4

# Minimal decode test: 64 tokens prefill, 32 tokens generate, batch=1
COMMON="-c 4096 -b 512 -ub 256 -npp 64 -ntg 32 -npl 1 --output-format jsonl"

echo "=========================================="
echo "Quick Test: Qwen3-4B, 4NUMA, 96 threads"
echo "=========================================="

# Group A: Fork-Join Pure
echo ""
echo ">>> Group A: Fork-Join Pure (no TP)"
$NUMA $FJ_BIN -m $MODEL -t $THREADS $COMMON --numa distribute \
    2>&1 | tee "$OUTDIR/fj_pure.log"

echo ""
echo ">>> Group B: Fork-Join + TP4"
$NUMA $FJ_BIN -m $MODEL -t $THREADS $COMMON \
    --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute \
    2>&1 | tee "$OUTDIR/fj_tp4.log"

echo ""
echo ">>> Group C: TaskInfer (Taskflow + TP4)"
$NUMA $TF_BIN -m $MODEL -t $THREADS $COMMON \
    --cpu-tp $TP --cpu-strategy taskflow --numa distribute \
    2>&1 | tee "$OUTDIR/task_tp4.log"

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="

# Show summary
echo ""
echo "=== Summary ==="
for f in "$OUTDIR"/*.log; do
    echo ""
    echo "--- $(basename $f) ---"
    # Extract the throughput line from JSONL or raw output
    grep -E "(S_PP|S_TG|t_pp|t_tg|\"pp\"|\"tg\")" "$f" 2>/dev/null | tail -3
done
