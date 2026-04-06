#!/bin/bash
# =================================================================
# exp1_run_14b.sh — Supplement: Qwen2.5-14B on all NUMA configs
# Same structure as main experiment
# =================================================================
set -euo pipefail

FJ_BIN=/home/haibin/tp/fork-join/build/bin/llama-batched-bench
TF_BIN=/home/haibin/tp/tp-llama/build/bin/llama-batched-bench

OUTDIR=/home/haibin/tp/exp1_results/raw
LOGFILE=/home/haibin/tp/exp1_results/exp1_14b.log
mkdir -p "$OUTDIR"

MODEL=$HOME/model/Qwen2.5-14B-Instruct-f16.gguf
MODEL_NAME=qwen25_14b

REPEATS=3
COMMON="-c 4096 -b 512 -ub 256 --output-format jsonl"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"
}

run_bench() {
    local label="$1"
    local outfile="$2"
    shift 2

    log "  RUN: $label -> $(basename $outfile)"

    if "$@" > "$outfile" 2>&1; then
        local speeds=$(grep "^{" "$outfile" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line.strip())
    print(f'  B={d[\"pl\"]}: PP={d[\"speed_pp\"]:.1f} TG={d[\"speed_tg\"]:.2f} t/s')
" 2>/dev/null)
        log "$speeds"
    else
        log "  FAILED: exit $?"
    fi
}

log "=========================================="
log "Experiment 1 Supplement: Qwen2.5-14B (F16)"
log "Model: $MODEL"
log "Start: $(date)"
log "=========================================="

for repeat in $(seq 1 $REPEATS); do
    log ""
    log "############## REPEAT $repeat/$REPEATS ##############"

    # ============================================================
    # 1 NUMA (node 0), 24 threads, no TP
    # ============================================================
    NUMA="numactl --cpunodebind=0 --membind=0"
    THREADS=24

    log ""
    log "====== 1NUMA, 24 threads ======"

    # Decode
    run_bench "1N FJ-Pure decode" \
        "$OUTDIR/${MODEL_NAME}_1n_fj_decode_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 64 -npl 1 $COMMON --numa distribute

    run_bench "1N TaskInfer decode" \
        "$OUTDIR/${MODEL_NAME}_1n_task_decode_r${repeat}.jsonl" \
        $NUMA $TF_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 64 -npl 1 $COMMON --cpu-strategy taskflow --numa distribute

    # Prefill
    run_bench "1N FJ-Pure prefill" \
        "$OUTDIR/${MODEL_NAME}_1n_fj_prefill_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 128,256,512 -ntg 1 -npl 1 $COMMON --numa distribute

    run_bench "1N TaskInfer prefill" \
        "$OUTDIR/${MODEL_NAME}_1n_task_prefill_r${repeat}.jsonl" \
        $NUMA $TF_BIN -m "$MODEL" -t $THREADS \
        -npp 128,256,512 -ntg 1 -npl 1 $COMMON --cpu-strategy taskflow --numa distribute

    # Batched Decode
    run_bench "1N FJ-Pure batch" \
        "$OUTDIR/${MODEL_NAME}_1n_fj_batch_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON --numa distribute

    run_bench "1N TaskInfer batch" \
        "$OUTDIR/${MODEL_NAME}_1n_task_batch_r${repeat}.jsonl" \
        $NUMA $TF_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON --cpu-strategy taskflow --numa distribute

    # ============================================================
    # 4 NUMA (nodes 0-3), 96 threads, TP=4
    # ============================================================
    NUMA="numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3"
    THREADS=96
    TP=4

    log ""
    log "====== 4NUMA, 96 threads, TP${TP} ======"

    # Decode
    run_bench "4N FJ-Pure decode" \
        "$OUTDIR/${MODEL_NAME}_4n_fj_decode_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 64 -npl 1 $COMMON --numa distribute

    run_bench "4N FJ+TP${TP} decode" \
        "$OUTDIR/${MODEL_NAME}_4n_fjtp_decode_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 64 -npl 1 $COMMON \
        --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

    run_bench "4N TaskInfer decode" \
        "$OUTDIR/${MODEL_NAME}_4n_task_decode_r${repeat}.jsonl" \
        $NUMA $TF_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 64 -npl 1 $COMMON \
        --cpu-tp $TP --cpu-strategy taskflow --numa distribute

    # Prefill
    run_bench "4N FJ-Pure prefill" \
        "$OUTDIR/${MODEL_NAME}_4n_fj_prefill_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 128,256,512 -ntg 1 -npl 1 $COMMON --numa distribute

    run_bench "4N FJ+TP${TP} prefill" \
        "$OUTDIR/${MODEL_NAME}_4n_fjtp_prefill_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 128,256,512 -ntg 1 -npl 1 $COMMON \
        --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

    run_bench "4N TaskInfer prefill" \
        "$OUTDIR/${MODEL_NAME}_4n_task_prefill_r${repeat}.jsonl" \
        $NUMA $TF_BIN -m "$MODEL" -t $THREADS \
        -npp 128,256,512 -ntg 1 -npl 1 $COMMON \
        --cpu-tp $TP --cpu-strategy taskflow --numa distribute

    # Batched Decode
    run_bench "4N FJ-Pure batch" \
        "$OUTDIR/${MODEL_NAME}_4n_fj_batch_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON --numa distribute

    run_bench "4N FJ+TP${TP} batch" \
        "$OUTDIR/${MODEL_NAME}_4n_fjtp_batch_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON \
        --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

    run_bench "4N TaskInfer batch" \
        "$OUTDIR/${MODEL_NAME}_4n_task_batch_r${repeat}.jsonl" \
        $NUMA $TF_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON \
        --cpu-tp $TP --cpu-strategy taskflow --numa distribute

    # ============================================================
    # 8 NUMA (all nodes), 192 threads, TP=8
    # ============================================================
    NUMA="numactl --cpunodebind=0,1,2,3,4,5,6,7 --membind=0,1,2,3,4,5,6,7"
    THREADS=192
    TP=8

    log ""
    log "====== 8NUMA, 192 threads, TP${TP} ======"

    # Decode
    run_bench "8N FJ-Pure decode" \
        "$OUTDIR/${MODEL_NAME}_8n_fj_decode_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 64 -npl 1 $COMMON --numa distribute

    run_bench "8N FJ+TP${TP} decode" \
        "$OUTDIR/${MODEL_NAME}_8n_fjtp_decode_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 64 -npl 1 $COMMON \
        --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

    run_bench "8N TaskInfer decode" \
        "$OUTDIR/${MODEL_NAME}_8n_task_decode_r${repeat}.jsonl" \
        $NUMA $TF_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 64 -npl 1 $COMMON \
        --cpu-tp $TP --cpu-strategy taskflow --numa distribute

    # Prefill
    run_bench "8N FJ-Pure prefill" \
        "$OUTDIR/${MODEL_NAME}_8n_fj_prefill_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 128,256,512 -ntg 1 -npl 1 $COMMON --numa distribute

    run_bench "8N FJ+TP${TP} prefill" \
        "$OUTDIR/${MODEL_NAME}_8n_fjtp_prefill_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 128,256,512 -ntg 1 -npl 1 $COMMON \
        --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

    run_bench "8N TaskInfer prefill" \
        "$OUTDIR/${MODEL_NAME}_8n_task_prefill_r${repeat}.jsonl" \
        $NUMA $TF_BIN -m "$MODEL" -t $THREADS \
        -npp 128,256,512 -ntg 1 -npl 1 $COMMON \
        --cpu-tp $TP --cpu-strategy taskflow --numa distribute

    # Batched Decode
    run_bench "8N FJ-Pure batch" \
        "$OUTDIR/${MODEL_NAME}_8n_fj_batch_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON --numa distribute

    run_bench "8N FJ+TP${TP} batch" \
        "$OUTDIR/${MODEL_NAME}_8n_fjtp_batch_r${repeat}.jsonl" \
        $NUMA $FJ_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON \
        --cpu-tp $TP --cpu-tp-runtime forkjoin --numa distribute

    run_bench "8N TaskInfer batch" \
        "$OUTDIR/${MODEL_NAME}_8n_task_batch_r${repeat}.jsonl" \
        $NUMA $TF_BIN -m "$MODEL" -t $THREADS \
        -npp 64 -ntg 32 -npl 1,4,8,16 $COMMON \
        --cpu-tp $TP --cpu-strategy taskflow --numa distribute
done

log ""
log "=========================================="
log "14B EXPERIMENT COMPLETE: $(date)"
log "=========================================="
