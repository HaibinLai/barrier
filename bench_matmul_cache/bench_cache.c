/*
 * bench_cache.c — Matmul kernel microbenchmark: cache staircase effect
 *
 * Measures the throughput (GFLOPS) of a single-core matmul kernel as a
 * function of the tile working-set size.  By sweeping the number of
 * output rows (b0) while keeping K fixed, the working set crosses L1d,
 * L2, and LLC boundaries, producing a visible "staircase" in throughput.
 *
 * This directly calls ggml_vec_dot_f16 — the same kernel used in
 * ggml_compute_forward_mul_mat_one_chunk — so the numbers reflect
 * production behavior.
 *
 * Build (from repo root):
 *   gcc -O2 -march=native -o bench_cache bench_matmul_cache/bench_cache.c \
 *       -I exp_gran_forkjoin/llama.cpp/ggml/include \
 *       -I exp_gran_forkjoin/llama.cpp/ggml/src/ggml-cpu \
 *       -L exp_gran_forkjoin/llama.cpp/build/bin \
 *       -lggml-cpu -lggml-base -lm \
 *       -Wl,-rpath,exp_gran_forkjoin/llama.cpp/build/bin
 *
 * Run:
 *   numactl --cpunodebind=0 --membind=0 ./bench_cache
 *
 * Output: CSV to stdout (pipe to file), human-readable to stderr.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "ggml.h"
#include "ggml-cpu.h"

/* ---------- vec.h is internal; declare what we need ---------- */
extern void ggml_vec_dot_f16(int n, float * restrict s, size_t bs,
                             ggml_fp16_t * restrict x, size_t bx,
                             ggml_fp16_t * restrict y, size_t by,
                             int nrc);

/* ---------- Configuration ---------- */
#define K         2560          /* inner dimension (Qwen3-4B FFN) */
#define N_WARMUP  20            /* warmup iterations */
#define N_MEASURE 200           /* measurement iterations */
#define N_SRC1    1             /* single column (TG-like) */

/* Time helper: returns nanoseconds */
static inline int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* Allocate aligned FP16 buffer filled with small random values */
static ggml_fp16_t * alloc_fp16(size_t n) {
    ggml_fp16_t * buf;
    if (posix_memalign((void **)&buf, 64, n * sizeof(ggml_fp16_t)) != 0) {
        fprintf(stderr, "alloc_fp16: posix_memalign failed\n");
        exit(1);
    }
    /* Fill with small random FP16 values to avoid denormals */
    srand(42);
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        buf[i] = ggml_fp32_to_fp16(v);
    }
    return buf;
}

/*
 * Run the benchmark for a given b0 (number of src0 rows = output rows).
 *
 * Simulates one "task" of ggml_compute_forward_mul_mat_one_chunk:
 *   for each of b0 rows:  vec_dot(K, &dst[row], ..., src0_row, ..., src1_col, ..., 1)
 *
 * Working set  ≈  b0 * K * 2  (weight tile, FP16)
 *              +  K * 2        (src1 column, reused across rows)
 *              +  b0 * 4       (dst, FP32)
 *
 * Returns: median latency in nanoseconds for one complete tile computation.
 */
static double bench_tile(int b0, ggml_fp16_t *src0, ggml_fp16_t *src1,
                         float *dst, int64_t *latencies) {
    const size_t row_bytes = (size_t)K * sizeof(ggml_fp16_t);

    /* Warmup — also pulls data into whatever cache level it fits */
    for (int t = 0; t < N_WARMUP; t++) {
        for (int r = 0; r < b0; r++) {
            ggml_vec_dot_f16(K, &dst[r], 0,
                             src0 + (size_t)r * K, 0,
                             src1, 0, 1);
        }
    }

    /* Measure */
    for (int t = 0; t < N_MEASURE; t++) {
        int64_t t0 = now_ns();
        for (int r = 0; r < b0; r++) {
            ggml_vec_dot_f16(K, &dst[r], 0,
                             src0 + (size_t)r * K, 0,
                             src1, 0, 1);
        }
        int64_t t1 = now_ns();
        latencies[t] = t1 - t0;
    }

    /* Sort for median */
    for (int i = 0; i < N_MEASURE - 1; i++) {
        for (int j = i + 1; j < N_MEASURE; j++) {
            if (latencies[j] < latencies[i]) {
                int64_t tmp = latencies[i];
                latencies[i] = latencies[j];
                latencies[j] = tmp;
            }
        }
    }

    return (double)latencies[N_MEASURE / 2];
}

int main(void) {
    /* b0 sweep: logarithmic-ish spacing from 1 to 512 */
    int b0_values[] = {
        1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32,
        40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512
    };
    int n_points = sizeof(b0_values) / sizeof(b0_values[0]);

    /* Allocate max-size buffers */
    int max_b0 = 512;
    ggml_fp16_t * src0 = alloc_fp16((size_t)max_b0 * K);  /* weight tile */
    ggml_fp16_t * src1 = alloc_fp16((size_t)K);            /* single src1 column */
    float * dst = (float *)calloc(max_b0, sizeof(float));
    int64_t * latencies = (int64_t *)malloc(N_MEASURE * sizeof(int64_t));

    if (!dst || !latencies) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* CSV header to stdout */
    printf("b0,K,working_set_kb,median_ns,gflops,ns_per_row\n");

    fprintf(stderr, "Matmul kernel cache microbenchmark\n");
    fprintf(stderr, "K=%d, N_WARMUP=%d, N_MEASURE=%d\n", K, N_WARMUP, N_MEASURE);
    fprintf(stderr, "%-6s  %-10s  %-12s  %-10s  %-10s\n",
            "b0", "WS (KB)", "median (us)", "GFLOPS", "ns/row");
    fprintf(stderr, "------  ----------  ------------  ----------  ----------\n");

    for (int i = 0; i < n_points; i++) {
        int b0 = b0_values[i];

        /* Working set: weight tile + src1 vector + dst */
        double ws_kb = ((double)b0 * K * 2 + K * 2 + b0 * 4) / 1024.0;

        double med_ns = bench_tile(b0, src0, src1, dst, latencies);

        /* GFLOPS = 2 * b0 * K / median_ns (ns -> seconds, then /1e9 for G) */
        double flops = 2.0 * b0 * K;
        double gflops = flops / med_ns;  /* med_ns in ns, flops/ns = GFLOPS */

        double ns_per_row = med_ns / b0;

        /* CSV to stdout */
        printf("%d,%d,%.2f,%.0f,%.4f,%.2f\n",
               b0, K, ws_kb, med_ns, gflops, ns_per_row);

        /* Human-readable to stderr */
        fprintf(stderr, "%-6d  %-10.1f  %-12.1f  %-10.4f  %-10.1f\n",
                b0, ws_kb, med_ns / 1000.0, gflops, ns_per_row);
    }

    free(src0);
    free(src1);
    free(dst);
    free(latencies);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
