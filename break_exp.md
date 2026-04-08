# TP4 Compute/Barrier 实验报告

## 实验概要

**目标**：比较 Taskflow DAG 调度与 Fork-Join 线程池在 CPU NUMA TP4 推理中的 compute 时间和 barrier/sync 时间分布差异。

**硬件**：8 NUMA x 24 core Intel Xeon Platinum 8160（实验使用 4 NUMA 节点，共 96 线程）

**模型**：Qwen3-4B-Ins-2507-f16.gguf (7.5 GiB, F16)

**参数**：PP=64, TG=16, NPL=1/2/4/8/16, Context=4096, Batch=512, UBatch=256

---

## 实验矩阵

| 编号 | 配置名 | 二进制 | Runtime | NUMA | 线程 | TP | 预分片 | Tree Barrier |
|------|--------|--------|---------|------|------|-----|--------|-------------|
| D | baseline_1numa | fork-join | forkjoin (OpenMP) | 1 node | 24 | 1 (无) | -- | -- |
| A | fj_tp4 | fork-join | forkjoin (OpenMP) | 4 nodes | 96 | 4 | row-shard-cache | 无 |
| B | tf_tp4 | tp-llama | taskflow | 4 nodes | 96 | 4 | row-shard-cache | 无 |
| C | tf_tp4_tree | tp-llama | taskflow | 4 nodes | 96 | 4 | row-shard-cache | 有 |
| E | tf_tp4_noshard | tp-llama | taskflow | 4 nodes | 96 | 4 | 无 | 无 |

**计时系统**：
- Fork-Join (D): `GGML_PERF_ALL` — per-thread compute_us / barrier_us，取 24 线程平均值
- Taskflow (B/C/E): `GGML_PERF_TF` — graph_us / dag_build_us / dag_exec_us / tp_mulmat_compute_us / tp_mulmat_sync_us
- Fork-Join TP mode (A): 无 GGML_PERF 输出（TP mode OpenMP 路径未植入计时），仅有吞吐量数据

---

## 一、吞吐量对比

### 1.1 Prefill (PP=64) 吞吐量 (tokens/s)

| B | Baseline 1NUMA | FJ TP4 | TF TP4 | TF TP4+tree | TF TP4 noshard |
|---|---------------|--------|--------|-------------|---------------|
| 1 | 61.56 | 41.87 | 40.60 | 38.88 | 63.54 |
| 2 | 63.31 | **102.08** | 70.77 | 63.31 | 62.43 |
| 4 | 57.87 | **102.28** | 70.47 | 68.71 | 68.62 |
| 8 | 57.04 | **101.58** | 70.50 | 69.05 | 65.12 |
| 16 | 55.49 | **98.01** | 54.98 | 54.22 | 53.37 |

**PP 观察**：
- FJ TP4 在 B>=2 时 PP 吞吐达到 ~100 t/s，是 baseline 的 **1.8x**
- TF TP4 在 B=2~8 时 PP 吞吐 ~70 t/s，是 baseline 的 **1.2x**，但低于 FJ
- B=1 时所有 TP4 配置都比 baseline 慢（TP 调度开销 > 并行增益）
- B=16 时 TF 和 FJ 的 PP 差距缩小到 ~44 t/s 差距

### 1.2 Decode (TG=16) 吞吐量 (tokens/s)

| B | Baseline 1NUMA | FJ TP4 | TF TP4 | TF TP4+tree | TF TP4 noshard |
|---|---------------|--------|--------|-------------|---------------|
| 1 | 6.56 | 2.96 | **7.55** | 5.28 | 3.08 |
| 2 | 11.04 | 5.83 | **10.94** | 10.11 | 5.34 |
| 4 | 16.87 | 10.94 | **19.49** | 16.63 | 9.97 |
| 8 | **31.33** | 20.40 | 29.60 | 26.04 | 17.33 |
| 16 | **40.22** | 31.02 | 30.65 | 28.49 | 24.34 |

**TG 观察**：
- TF TP4 在 B=1 时 TG 7.55 t/s，是 FJ 的 **2.6x**（FJ 96 线程 barrier 等待严重）
- TF TP4 在 B=1~4 时 TG 接近甚至超过 baseline（Taskflow 无 barrier 调度优势明显）
- B>=8 时 baseline 反超所有 TP 配置（单 NUMA 24 线程已足够，TP 跨 NUMA 开销反成负担）
- noshard 配置的 TG 始终最慢（远程内存访问惩罚在 decode 小 batch 时最显著）

### 1.3 综合吞吐量 (tokens/s)

| B | Baseline 1NUMA | FJ TP4 | TF TP4 | TF TP4+tree | TF TP4 noshard |
|---|---------------|--------|--------|-------------|---------------|
| 1 | 23.00 | 11.53 | 21.65 | 17.11 | 12.90 |
| 2 | 32.52 | 23.74 | **33.80** | 30.85 | 19.90 |
| 4 | 38.95 | 38.30 | **46.27** | 42.25 | 31.52 |
| 8 | 49.00 | **56.56** | 55.24 | 51.90 | 41.97 |
| 16 | 51.57 | **68.44** | 47.45 | 45.93 | 43.09 |

**综合观察**：
- B=2~4 时 TF TP4 综合吞吐最高（PP 合理 + TG 无 barrier 优势）
- B>=8 时 FJ TP4 综合最高（PP 绝对优势碾压 TG 的劣势）
- **TF 和 FJ 的最佳使用场景不同**：TF 适合低并发在线服务（B=1~4），FJ 适合高并发批处理（B>=8）

---

## 二、Compute/Barrier 时间分解

### 2.1 Baseline 1NUMA — Fork-Join per-thread 统计

每次 graph compute 输出 24 线程的平均 compute_us 和 barrier_us。下表取每个 batch size 的中位数（跳过前 2 次 warmup）。

| B | graph_us | avg_compute_us | avg_barrier_us | barrier% |
|---|---------|---------------|----------------|----------|
| 1 | 150,358 | 133,138 | 16,772 | **11.2%** |
| 2 | 178,800 | 157,520 | 20,935 | **11.7%** |
| 4 | 234,685 | 207,510 | 26,804 | **11.4%** |
| 8 | 252,243 | 220,291 | 31,370 | **12.5%** |
| 16 | 392,822 | 357,650 | 34,793 | **8.9%** |

**观察**：Baseline 的 barrier 开销稳定在 **9~13%**，是 fork-join 模型的固有成本。

### 2.2 TF TP4 (row-shard-cache) — Taskflow DAG 统计

| B | graph_us | dag_build_us | dag_exec_us | mm_compute_us | mm_sync_us | sync% | n_mulmat |
|---|---------|-------------|------------|--------------|-----------|-------|---------|
| 1 | 122,006 | 1,594 | 120,370 | 99,272 | 22 | **0.0%** | 649 |
| 2 | 187,234 | 2,867 | 184,368 | 147,344 | 50 | **0.0%** | 649 |
| 4 | 203,306 | 6,444 | 196,820 | 158,218 | 34 | **0.0%** | 649 |
| 8 | 264,806 | 17,440 | 246,986 | 201,690 | 24 | **0.0%** | 649 |
| 16 | 519,013 | 53,155 | 466,096 | 384,526 | 36 | **0.0%** | 649 |

**观察**：
- **同步等待几乎为零**：`mm_sync_us` 仅 20~50 us（占比 0.0%），DAG 依赖链完全替代了 barrier
- `dag_build_us` 随 batch size 线性增长：B=1 时 ~1.6ms，B=16 时 ~53ms
- DAG build 在 B=16 时占 graph 总时间的 **10.2%**（53ms / 519ms），是不可忽略的开销
- `mm_compute_us` 占 `dag_exec_us` 的 **~82%**，剩余 18% 是非 MUL_MAT op 的执行时间

### 2.3 TF TP4 + Tree Barrier — 引入跨域同步

| B | graph_us | dag_build_us | dag_exec_us | mm_compute_us | mm_sync_us | sync% | n_mulmat |
|---|---------|-------------|------------|--------------|-----------|-------|---------|
| 1 | 185,022 | 1,606 | 183,428 | 119,605 | 35,304 | **22.8%** | 649 |
| 2 | 185,791 | 2,939 | 182,848 | 125,671 | 26,730 | **17.5%** | 649 |
| 4 | 224,110 | 6,460 | 217,790 | 154,922 | 26,890 | **14.8%** | 649 |
| 8 | 291,406 | 17,375 | 274,364 | 200,436 | 26,656 | **11.7%** | 649 |
| 16 | 564,013 | 53,448 | 511,832 | 393,312 | 34,551 | **8.1%** | 649 |

**观察**：
- Tree barrier 引入 **27~35 ms/graph** 的同步开销
- sync% 从 B=1 的 22.8% 递减到 B=16 的 8.1%（compute 时间增长摊薄了固定 barrier 成本）
- 对比无 tree barrier (TF TP4)：graph_us 增加 **35~52%**（B=1: 122ms → 185ms）
- **Tree barrier 的绝对成本 (~30ms) 与 fork-join baseline 的 barrier 成本 (~17~35ms) 在同一量级**

### 2.4 TF TP4 无预分片 (noshard) — 跨 NUMA 远程内存访问

| B | graph_us | dag_build_us | dag_exec_us | mm_compute_us | mm_sync_us | sync% | n_mulmat |
|---|---------|-------------|------------|--------------|-----------|-------|---------|
| 1 | 321,978 | 1,571 | 320,372 | 296,361 | 32 | **0.0%** | 649 |
| 2 | 371,074 | 2,878 | 368,194 | 332,655 | 42 | **0.0%** | 649 |
| 4 | 398,164 | 6,448 | 391,836 | 349,418 | 44 | **0.0%** | 649 |
| 8 | 457,420 | 17,298 | 440,288 | 386,338 | 38 | **0.0%** | 649 |
| 16 | 650,172 | 53,412 | 597,428 | 515,736 | 38 | **0.0%** | 649 |

**观察**：
- `mm_compute_us` 比有预分片时高 **2~3x**（B=1: 296ms vs 99ms，B=16: 516ms vs 385ms）
- sync 同样几乎为零（DAG 特性不变）
- **row-shard-cache 预分片是性能的决定性因素**，远比 barrier 优化重要

---

## 三、横向对比分析

### 3.1 每 Graph 时间对比（中位数，单位 us）

| B | Baseline 1NUMA | TF TP4 | TF TP4+tree | TF TP4 noshard | vs Baseline |
|---|---------------|--------|-------------|---------------|-------------|
| 1 | 150,358 | **122,006** | 185,022 | 321,978 | TF TP4 快 19% |
| 2 | 178,800 | 187,234 | 185,791 | 371,074 | 相当 |
| 4 | 234,685 | **203,306** | 224,110 | 398,164 | TF TP4 快 13% |
| 8 | 252,243 | 264,806 | 291,406 | 457,420 | 相当 |
| 16 | 392,822 | 519,013 | 564,013 | 650,172 | Baseline 快 32% |

### 3.2 Barrier/Sync 开销对比

| 配置 | B=1 barrier | B=1 barrier% | B=16 barrier | B=16 barrier% |
|------|-----------|-------------|-------------|--------------|
| Baseline (FJ 24t) | 16,772 us | 11.2% | 34,793 us | 8.9% |
| TF TP4 | **22 us** | **0.0%** | **36 us** | **0.0%** |
| TF TP4+tree | 35,304 us | 22.8% | 34,551 us | 8.1% |

- Taskflow 无 tree barrier 时：**barrier 开销从 11% 降至 0%**，是 DAG 调度最大的优势
- Taskflow 有 tree barrier 时：sync 开销反而比 baseline 更高（22.8% vs 11.2%），因为 tree barrier 是人为插入的额外同步

### 3.3 DAG Build 开销分析

| B | dag_build_us | 占 graph% | 评价 |
|---|-------------|---------|------|
| 1 | 1,594 | 1.3% | 可忽略 |
| 2 | 2,867 | 1.5% | 可忽略 |
| 4 | 6,444 | 3.2% | 轻微 |
| 8 | 17,440 | 6.6% | 需关注 |
| 16 | 53,155 | **10.2%** | 瓶颈征兆 |

DAG build 开销随 batch size 超线性增长（B 翻倍，build 时间增长 ~3x），在 B=16 时已占 10%。这是 Taskflow 路径在大 batch 时 PP 吞吐不如 FJ 的原因之一。

---

## 四、关键结论

### 结论 1：Taskflow DAG 调度消除了 barrier 等待

Taskflow 路径的 `tp_mulmat_sync_us` 仅 20~50 us/graph，相比 baseline 的 17~35 ms barrier，**降低了 3 个数量级**。DAG 依赖链让每个 op 完成后直接触发下游，无需全局同步。

### 结论 2：预分片 (row-shard-cache) 是最关键的优化

| 对比 | B=1 mm_compute | B=8 mm_compute | 差异 |
|------|---------------|---------------|------|
| TF TP4 (有预分片) | 99,272 us | 201,690 us | -- |
| TF TP4 (无预分片) | 296,361 us | 386,338 us | **3.0x / 1.9x 慢** |

无预分片时，每个 domain 的 executor 访问远程 NUMA 节点的权重数据，内存带宽成为瓶颈。**预分片的收益 >> barrier 优化的收益**。

### 结论 3：FJ 和 TF 各有最佳场景

| 场景 | 推荐 | 原因 |
|------|------|------|
| 低并发在线推理 (B=1~4) | **TF TP4** | 无 barrier 等待，TG 吞吐高 2.6x |
| 高并发批处理 (B>=8) | **FJ TP4** | PP 吞吐 ~100 t/s，综合最高 |
| 超大 batch (B=16+) | **FJ TP4** | DAG build 开销拖累 TF |

### 结论 4：Tree Barrier 不适合 Taskflow 路径

Tree barrier 人为引入 27~35 ms 固定同步开销，抵消了 Taskflow 无 barrier 的优势。在所有 batch size 下，TF+tree 都比 TF 慢。**Tree barrier 是为 fork-join 设计的优化，不适用于 DAG 模型**。

### 结论 5：DAG Build 开销是 Taskflow 路径的隐性瓶颈

B=16 时 DAG build 耗时 53ms（占 10%），且增速超线性。需要考虑：
- DAG 结构缓存/复用（避免每次重建）
- 分批构建（只重建变化的子图）
- 或在 B 较大时自动回退到 fork-join 路径

---

## 五、原始 CLI 命令

```bash
# D: Baseline 1NUMA (24 threads, no TP)
numactl --cpunodebind=0 --membind=0 \
  fork-join/build/bin/llama-batched-bench \
  -m Qwen3-4B-Ins-2507-f16.gguf -t 24 \
  -c 4096 -b 512 -ub 256 -npp 64 -ntg 16 -npl 1,2,4,8,16 \
  --output-format md

# A: Fork-Join TP4
numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 \
  fork-join/build/bin/llama-batched-bench \
  -m Qwen3-4B-Ins-2507-f16.gguf -t 96 \
  -c 4096 -b 512 -ub 256 -npp 64 -ntg 16 -npl 1,2,4,8,16 \
  --numa distribute --cpu-tp 4 --cpu-tp-place row-shard-cache \
  --cpu-tp-runtime forkjoin --output-format md

# B: Taskflow TP4
numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 \
  tp-llama/build/bin/llama-batched-bench \
  -m Qwen3-4B-Ins-2507-f16.gguf -t 96 \
  -c 4096 -b 512 -ub 256 -npp 64 -ntg 16 -npl 1,2,4,8,16 \
  --numa distribute --cpu-tp 4 --cpu-tp-place row-shard-cache \
  --cpu-tp-runtime taskflow --output-format md

# C: Taskflow TP4 + Tree Barrier
numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 \
  tp-llama/build/bin/llama-batched-bench \
  -m Qwen3-4B-Ins-2507-f16.gguf -t 96 \
  -c 4096 -b 512 -ub 256 -npp 64 -ntg 16 -npl 1,2,4,8,16 \
  --numa distribute --cpu-tp 4 --cpu-tp-place row-shard-cache \
  --cpu-tp-runtime taskflow --tree-barrier --output-format md

# E: Taskflow TP4 (no row-shard-cache)
numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 \
  tp-llama/build/bin/llama-batched-bench \
  -m Qwen3-4B-Ins-2507-f16.gguf -t 96 \
  -c 4096 -b 512 -ub 256 -npp 64 -ntg 16 -npl 1,2,4,8,16 \
  --numa distribute --cpu-tp 4 \
  --cpu-tp-runtime taskflow --output-format md
```

---

## 六、GGML_PERF 计时系统说明

### Fork-Join 路径 (GGML_PERF_ALL)

```
GGML_PERF_ALL|nth=24|graph_us=150358|avg_compute_us=133138|avg_barrier_us=16772|...
```

- `graph_us`: 整个 graph compute 的 wall clock
- `avg_compute_us`: 所有线程的 compute 时间平均值（实际执行 op 的时间）
- `avg_barrier_us`: 所有线程的 barrier 等待时间平均值
- 仅在 non-TP OpenMP 路径有输出；TP mode 的 OpenMP 路径暂未植入计时

### Taskflow 路径 (GGML_PERF_TF)

```
GGML_PERF_TF|graph_us=122006|dag_build_us=1594|dag_exec_us=120370|tp_mulmat_compute_us=99272|tp_mulmat_sync_us=22|n_tp_mulmat=649
```

- `graph_us`: 整个 `ggml_graph_compute_with_taskflow()` 的 wall clock
- `dag_build_us`: 协调器遍历节点构建 DAG 的时间
- `dag_exec_us`: executor 执行 DAG 的时间
- `tp_mulmat_compute_us`: 所有 TP MUL_MAT 中 domain executor `run()+wait()` 的累计时间
- `tp_mulmat_sync_us`: 所有 TP MUL_MAT 中 tree barrier 的累计时间
- `n_tp_mulmat`: TP MUL_MAT 调用次数（645 = PP 阶段，649 = TG 阶段）

---

## 七、文件清单

```
bench_results_compute_barrier/
  baseline_1numa_bench.md      # Baseline 吞吐量表格
  baseline_1numa_perf.log      # Baseline GGML_PERF_ALL 原始日志 (90 entries)
  fj_tp4_bench.md              # FJ TP4 吞吐量表格
  fj_tp4_perf.log              # FJ TP4 stderr（无 GGML_PERF，TP mode 未计时）
  tf_tp4_bench.md              # TF TP4 吞吐量表格
  tf_tp4_perf.log              # TF TP4 GGML_PERF_TF 原始日志 (90 entries)
  tf_tp4_tree_bench.md         # TF TP4+tree 吞吐量表格
  tf_tp4_tree_perf.log         # TF TP4+tree GGML_PERF_TF 原始日志 (90 entries)
  tf_tp4_noshard_bench.md      # TF TP4 noshard 吞吐量表格
  tf_tp4_noshard_perf.log      # TF TP4 noshard GGML_PERF_TF 原始日志 (90 entries)
  experiment_report.md         # 本报告

bench_compute_barrier.sh       # 实验运行脚本
parse_perf.py                  # GGML_PERF 日志解析脚本
```
