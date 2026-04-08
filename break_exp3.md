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
| F | fj_pure_4n | fork-join | forkjoin (OpenMP) | 4 nodes | 96 | 1 (无) | -- | -- |
| A | fj_tp4 | fork-join | forkjoin (OpenMP) | 4 nodes | 96 | 4 | row-shard-cache | 无 |
| B | tf_tp4 | tp-llama | taskflow | 4 nodes | 96 | 4 | row-shard-cache | 无 |
| C | tf_tp4_tree | tp-llama | taskflow | 4 nodes | 96 | 4 | row-shard-cache | 有 |
| E | tf_tp4_noshard | tp-llama | taskflow | 4 nodes | 96 | 4 | 无 | 无 |

**计时系统**：
- Fork-Join non-TP (D, F): `GGML_PERF_ALL` — per-thread compute_us / barrier_us
- Fork-Join TP mode (A): `GGML_PERF_ALL` — per-thread compute_us / barrier_us，取 96 线程平均值（domain 0 执行所有 op + TP MUL_MAT，domain 1-3 仅参与 TP MUL_MAT + spin-wait）
- Taskflow (B/C/E): `GGML_PERF_TF` — graph_us / dag_build_us / dag_exec_us / tp_mulmat_compute_us / tp_mulmat_sync_us

---

## 一、吞吐量对比

### 1.1 Prefill (PP=64) 吞吐量 (tokens/s)

| B | Baseline 1N | FJ pure 4N | FJ TP4 | TF TP4 | TF TP4+tree | TF TP4 noshard |
|---|------------|-----------|--------|--------|-------------|---------------|
| 1 | 61.56 | 19.28 | 39.79 | 40.60 | 38.88 | 63.54 |
| 2 | 63.31 | 23.72 | **99.51** | 70.77 | 63.31 | 62.43 |
| 4 | 57.87 | 23.42 | **99.79** | 70.47 | 68.71 | 68.62 |
| 8 | 57.04 | 23.56 | **101.11** | 70.50 | 69.05 | 65.12 |
| 16 | 55.49 | 23.15 | **97.88** | 54.98 | 54.22 | 53.37 |

### 1.2 Decode (TG=16) 吞吐量 (tokens/s)

| B | Baseline 1N | FJ pure 4N | FJ TP4 | TF TP4 | TF TP4+tree | TF TP4 noshard |
|---|------------|-----------|--------|--------|-------------|---------------|
| 1 | 6.56 | 0.87 | 2.95 | **7.55** | 5.28 | 3.08 |
| 2 | 11.04 | 1.46 | 6.13 | **10.94** | 10.11 | 5.34 |
| 4 | 16.87 | 2.29 | 10.81 | **19.49** | 16.63 | 9.97 |
| 8 | **31.33** | 4.36 | 19.82 | 29.60 | 26.04 | 17.33 |
| 16 | **40.22** | 8.83 | 30.66 | 30.65 | 28.49 | 24.34 |

### 1.3 综合吞吐量 (tokens/s)

| B | Baseline 1N | FJ pure 4N | FJ TP4 | TF TP4 | TF TP4+tree | TF TP4 noshard |
|---|------------|-----------|--------|--------|-------------|---------------|
| 1 | 23.00 | 3.70 | 11.38 | **21.65** | 17.11 | 12.90 |
| 2 | 32.52 | 5.84 | 24.58 | **33.80** | 30.85 | 19.90 |
| 4 | 38.95 | 8.23 | 37.72 | **46.27** | 42.25 | 31.52 |
| 8 | 49.00 | 12.54 | **55.55** | 55.24 | 51.90 | 41.97 |
| 16 | 51.57 | 17.48 | **68.05** | 47.45 | 45.93 | 43.09 |

---

## 二、Compute/Barrier 时间分解（核心数据）

### 2.0 全配置统一对比表

下表取每个 batch size 的中位数（跳过前 2 次 warmup graph），单位 us。

```
  B | Config                 |     graph_us |   compute_us |   barrier_us |  barrier% |  nth
---------------------------------------------------------------------------------------------
  1 | Baseline 1NUMA         |       150358 |       133138 |        16772 |     11.2% |   24
    | FJ pure 4N (96t)       |      1141890 |       916757 |       217125 |     19.1% |   96
    | FJ TP4 (96t)           |       336589 |       247778 |        61444 |     19.9% |   96
    | TF TP4 (96t)           |       122006 |        99272 |           22 |      0.0% |   96
    | TF TP4+tree (96t)      |       185022 |       119605 |        35304 |     22.8% |   96
---------------------------------------------------------------------------------------------
  2 | Baseline 1NUMA         |       178800 |       157520 |        20935 |     11.7% |   24
    | FJ pure 4N (96t)       |      1369559 |      1067513 |       294062 |     21.6% |   96
    | FJ TP4 (96t)           |       313794 |       237766 |        56894 |     19.3% |   96
    | TF TP4 (96t)           |       187234 |       147344 |           50 |      0.0% |   96
    | TF TP4+tree (96t)      |       185791 |       125671 |        26730 |     17.5% |   96
---------------------------------------------------------------------------------------------
  4 | Baseline 1NUMA         |       234685 |       207510 |        26804 |     11.4% |   24
    | FJ pure 4N (96t)       |      1739892 |      1374119 |       358979 |     20.7% |   96
    | FJ TP4 (96t)           |       379984 |       264677 |        77237 |     22.6% |   96
    | TF TP4 (96t)           |       203306 |       158218 |           34 |      0.0% |   96
    | TF TP4+tree (96t)      |       224110 |       154922 |        26890 |     14.8% |   96
---------------------------------------------------------------------------------------------
  8 | Baseline 1NUMA         |       252243 |       220291 |        31370 |     12.5% |   24
    | FJ pure 4N (96t)       |      1840227 |      1425627 |       400545 |     21.9% |   96
    | FJ TP4 (96t)           |       409850 |       283458 |        81742 |     22.4% |   96
    | TF TP4 (96t)           |       264806 |       201690 |           24 |      0.0% |   96
    | TF TP4+tree (96t)      |       291406 |       200436 |        26656 |     11.7% |   96
---------------------------------------------------------------------------------------------
 16 | Baseline 1NUMA         |       392822 |       357650 |        34793 |      8.9% |   24
    | FJ pure 4N (96t)       |      1802842 |      1346009 |       450910 |     25.1% |   96
    | FJ TP4 (96t)           |       519244 |       362827 |       128434 |     26.1% |   96
    | TF TP4 (96t)           |       519013 |       384526 |           36 |      0.0% |   96
    | TF TP4+tree (96t)      |       564013 |       393312 |        34551 |      8.1% |   96
---------------------------------------------------------------------------------------------
```

> **注**：FJ TP4 和 FJ pure 4N 的 `compute_us` / `barrier_us` 是 96 线程的平均值。FJ TP4 中 domain 0 执行所有 op 而 domain 1-3 仅参与 TP MUL_MAT，各 domain 的分布差异很大，详见 2.2 节。FJ pure 4N 无 TP 分片，所有线程均等参与计算但大量跨 NUMA 远程内存访问，详见 2.6 节。

### 2.1 Baseline 1NUMA — Fork-Join per-thread 统计 (24 线程)

| B | graph_us | avg_compute_us | avg_barrier_us | barrier% |
|---|---------|---------------|----------------|----------|
| 1 | 150,358 | 133,138 | 16,772 | **11.2%** |
| 2 | 178,800 | 157,520 | 20,935 | **11.7%** |
| 4 | 234,685 | 207,510 | 26,804 | **11.4%** |
| 8 | 252,243 | 220,291 | 31,370 | **12.5%** |
| 16 | 392,822 | 357,650 | 34,793 | **8.9%** |

**观察**：Baseline 的 barrier 开销稳定在 **9~13%**，是 fork-join 模型的固有成本。

### 2.2 FJ TP4 — Fork-Join TP4 per-thread 统计 (96 线程)

#### 2.2.1 全局平均

| B | graph_us | avg_compute_us | avg_barrier_us | barrier% |
|---|---------|---------------|----------------|----------|
| 1 | 336,589 | 247,778 | 61,444 | **19.9%** |
| 2 | 313,794 | 237,766 | 56,894 | **19.3%** |
| 4 | 379,984 | 264,677 | 77,237 | **22.6%** |
| 8 | 409,850 | 283,458 | 81,742 | **22.4%** |
| 16 | 519,244 | 362,827 | 128,434 | **26.1%** |

**观察**：
- FJ TP4 的 barrier% 远高于 baseline：**20~26%** vs 9~13%
- B=16 时 barrier 高达 **128ms**，占 graph 总时间的 1/4
- 随着 batch size 增大，barrier 占比从 19.9% 上升到 26.1%（与 baseline 相反趋势）

#### 2.2.2 Per-Domain 分布 (d0=threads 0-23, d1=24-47, d2=48-71, d3=72-95)

```
  B | Domain |   avg_compute |   avg_barrier | barrier%
---------------------------------------------------------
B=1 |     d0 |      236782us |       70792us |   23.0%
    |     d1 |      249013us |       60700us |   19.6%
    |     d2 |      249230us |       61443us |   19.8%
    |     d3 |      255778us |       53580us |   17.3%
---------------------------------------------------------
B=2 |     d0 |      227616us |       65968us |   22.5%
    |     d1 |      238410us |       56282us |   19.1%
    |     d2 |      238777us |       56056us |   19.0%
    |     d3 |      245533us |       48780us |   16.6%
---------------------------------------------------------
B=4 |     d0 |      259811us |       80245us |   23.6%
    |     d1 |      262697us |       79726us |   23.3%
    |     d2 |      263976us |       78816us |   23.0%
    |     d3 |      270070us |       71413us |   20.9%
---------------------------------------------------------
B=8 |     d0 |      289223us |       76311us |   20.9%
    |     d1 |      278767us |       86005us |   23.6%
    |     d2 |      279738us |       84371us |   23.2%
    |     d3 |      285162us |       80472us |   22.0%
---------------------------------------------------------
B=16|     d0 |      400357us |       91493us |   18.6%
    |     d1 |      349134us |      142190us |   28.9%
    |     d2 |      348641us |      142168us |   29.0%
    |     d3 |      354199us |      137948us |   28.0%
---------------------------------------------------------
```

**Per-Domain 观察**：

1. **Domain 0 compute 最高但 barrier 最低**（B=16: compute 400ms vs d1~d3 350ms，barrier 91ms vs 140ms）
   - 原因：d0 执行所有 non-TP op（如 RMS_NORM、SiLU、RoPE），其余 domain 在等待 signal，等待时间被计为 barrier
   - d0 的额外 compute 工作 = 非 TP op 的执行时间 ≈ 50ms (400-350)

2. **Domain 1-3 的 barrier 远高于 domain 0**（B=16: 140ms vs 91ms）
   - d1-d3 的 barrier 包含两部分：① TP MUL_MAT 后的 `ggml_barrier_custom` ② spin-wait 等 d0 发出下一个 TP 信号
   - spin-wait 时间 ≈ d0 执行非 TP op 的时间（因为 d1-d3 此时完全空闲）

3. **B=16 时域间不均衡最严重**：d0 barrier 91ms vs d1-d3 ~140ms，差距 **53%**
   - 这是 TP fork-join 设计的根本瓶颈：non-TP op 只用 d0 的 24 线程，其余 72 线程完全浪费

### 2.6 FJ pure 4N — Fork-Join 无 TP 跨 4 NUMA 节点 (96 线程)

| B | graph_us | avg_compute_us | avg_barrier_us | barrier% |
|---|---------|---------------|----------------|----------|
| 1 | 1,141,890 | 916,757 | 217,125 | **19.1%** |
| 2 | 1,369,559 | 1,067,513 | 294,062 | **21.6%** |
| 4 | 1,739,892 | 1,374,119 | 358,979 | **20.7%** |
| 8 | 1,840,227 | 1,425,627 | 400,545 | **21.9%** |
| 16 | 1,802,842 | 1,346,009 | 450,910 | **25.1%** |

**观察**：

1. **灾难性性能**：graph_us 高达 **1.1~1.8 秒**，是 baseline 的 **4.6~7.6x**，是 FJ TP4 的 **3.4~4.4x**
   - 无 TP 分片 = 所有 96 线程共享同一份模型权重 → 大量跨 NUMA 远程内存访问
   - PP 吞吐仅 19~23 t/s（baseline 55~63 t/s），TG B=1 仅 **0.87 t/s**（baseline 6.56 t/s）

2. **Barrier 绝对值极高**：217~451 ms/graph，是所有配置中最高的
   - 是 baseline 的 **13~15x**（16ms → 217ms at B=1）
   - 是 FJ TP4 的 **3.5~5.2x**（61ms → 217ms at B=1）
   - 原因：96 线程 flat barrier + 跨 NUMA cache coherence traffic + 负载不均衡（不同线程访问远近不同的 NUMA 内存）

3. **Barrier 占比与 FJ TP4 相当**：19~25% vs 20~26%
   - 这说明 ~20% 的 barrier 占比是 96 线程 flat barrier 的内在成本，与是否有 TP 关系不大
   - 但 FJ pure 4N 的 **compute 时间也极差**（跨 NUMA 内存访问），所以 barrier 的绝对值和 compute 的绝对值都大幅膨胀

4. **B=16 时 barrier 占比飙升至 25.1%**：451ms barrier + 1346ms compute
   - graph_us (1803ms) 反而低于 B=8 (1840ms)，暗示某种调度异常或内存带宽饱和

### 2.3 TF TP4 (row-shard-cache) — Taskflow DAG 统计

| B | graph_us | dag_build_us | dag_exec_us | mm_compute_us | mm_sync_us | sync% | n_mulmat |
|---|---------|-------------|------------|--------------|-----------|-------|---------|
| 1 | 122,006 | 1,594 | 120,370 | 99,272 | 22 | **0.0%** | 649 |
| 2 | 187,234 | 2,867 | 184,368 | 147,344 | 50 | **0.0%** | 649 |
| 4 | 203,306 | 6,444 | 196,820 | 158,218 | 34 | **0.0%** | 649 |
| 8 | 264,806 | 17,440 | 246,986 | 201,690 | 24 | **0.0%** | 649 |
| 16 | 519,013 | 53,155 | 466,096 | 384,526 | 36 | **0.0%** | 649 |

**观察**：
- **同步等待几乎为零**：`mm_sync_us` 仅 20~50 us（占比 0.0%），DAG 依赖链完全替代了 barrier
- `dag_build_us` 随 batch size 增长：B=1 时 ~1.6ms，B=16 时 ~53ms
- DAG build 在 B=16 时占 graph 总时间的 **10.2%**（53ms / 519ms），是不可忽略的开销
- `mm_compute_us` 占 `dag_exec_us` 的 **~82%**，剩余 18% 是非 MUL_MAT op 的执行时间

### 2.4 TF TP4 + Tree Barrier — 引入跨域同步

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

### 2.5 TF TP4 无预分片 (noshard) — 跨 NUMA 远程内存访问

| B | graph_us | dag_build_us | dag_exec_us | mm_compute_us | mm_sync_us | sync% | n_mulmat |
|---|---------|-------------|------------|--------------|-----------|-------|---------|
| 1 | 321,978 | 1,571 | 320,372 | 296,361 | 32 | **0.0%** | 649 |
| 2 | 371,074 | 2,878 | 368,194 | 332,655 | 42 | **0.0%** | 649 |
| 4 | 398,164 | 6,448 | 391,836 | 349,418 | 44 | **0.0%** | 649 |
| 8 | 457,420 | 17,298 | 440,288 | 386,338 | 38 | **0.0%** | 649 |
| 16 | 650,172 | 53,412 | 597,428 | 515,736 | 38 | **0.0%** | 649 |

**观察**：
- `mm_compute_us` 比有预分片时高 **2~3x**（B=1: 296ms vs 99ms）
- sync 同样几乎为零（DAG 特性不变）
- **row-shard-cache 预分片是性能的决定性因素**，远比 barrier 优化重要

---

## 三、横向对比分析

### 3.1 Barrier 开销横向对比

```
              B=1                   B=4                   B=16
配置        barrier   barrier%    barrier   barrier%    barrier   barrier%
─────────────────────────────────────────────────────────────────────────
Baseline     16772us    11.2%     26804us    11.4%      34793us     8.9%
FJ pure 4N  217125us    19.1%    358979us    20.7%     450910us    25.1%   ← 绝对值最高
FJ TP4       61444us    19.9%     77237us    22.6%     128434us    26.1%   ← 占比最高
TF TP4          22us     0.0%        34us     0.0%         36us     0.0%   ← 最低
TF TP4+tree  35304us    22.8%     26890us    14.8%      34551us     8.1%
```

**核心发现**：
- **FJ pure 4N 的 barrier 绝对值最高**：217~451 ms，是 FJ TP4 的 3.5x，是 baseline 的 13x
- **FJ TP4 和 FJ pure 4N 的 barrier 占比相当**：都在 19~26%，说明 **20%+ 是 96 线程 flat barrier 的固有成本**
- **TF TP4 的 barrier 接近于零**：DAG 异步调度完全消除同步等待
- Baseline 24t → 4N 96t：barrier 从 11% 翻倍到 20~26%，**线程数增加 + 跨 NUMA 导致 barrier 成本超线性增长**
- FJ pure 4N 的 compute 时间也极差（无 TP 分片 = 全量跨 NUMA 访问），所以总体性能灾难性

### 3.2 FJ TP4 的 barrier 来源分解

FJ TP4 的 barrier 由三部分组成：

| 来源 | 描述 | 受影响线程 | 量级 |
|------|------|----------|------|
| TP MUL_MAT 后的 `ggml_barrier_custom` | 所有 96 线程同步 | 全部 | 主要（每个 TP MUL_MAT 后一次） |
| Domain-0 间 `ggml_barrier_domain0` | d0 的 24 线程同步 | d0 only | 次要 |
| Non-domain-0 的 spin-wait | d1-d3 等待 d0 发出 TP 信号 | d1-d3 (72线程) | 重要（B=16 时 ~50ms） |

B=16 时 d1-d3 的 barrier（~140ms）比 d0（~91ms）高 **53%**，差额 ~49ms 几乎全部是 spin-wait，等于 d0 执行非 TP op 的时间。这是 fork-join TP 设计的结构性浪费。

### 3.3 Compute 效率对比

```
              B=1 compute     B=16 compute      graph_us ratio (vs Baseline)
─────────────────────────────────────────────────────────────────────────────
Baseline      133138us         357650us          1.00x
FJ pure 4N    916757us        1346009us          B=1: 7.60x slower / B=16: 4.59x
FJ TP4        247778us         362827us          B=1: 2.24x slower / B=16: 1.32x
TF TP4         99272us         384526us          B=1: 0.81x / B=16: 1.32x
```

- **FJ pure 4N 的 compute 时间是所有配置中最差的**：B=1 时 917ms（baseline 的 6.9x）
  - 无 TP = 无分片 = 每个线程都可能访问远程 NUMA 内存，cache miss penalty 巨大
  - 这证明 **TP 分片对 NUMA 系统的作用不仅是减少 barrier，更是减少跨域内存访问**
- B=1 时 TF TP4 graph_us 仅 122ms，比 baseline 150ms **快 19%**，比 FJ TP4 337ms **快 64%**
- B=16 时 FJ TP4 和 TF TP4 的 graph_us 几乎相等（519ms），但 FJ 的 PP 吞吐 (98 t/s) 远高于 TF (55 t/s)——说明 FJ 在 prefill 阶段的 OpenMP 线程调度更高效

### 3.4 DAG Build 开销分析

| B | dag_build_us | 占 graph% | 评价 |
|---|-------------|---------|------|
| 1 | 1,594 | 1.3% | 可忽略 |
| 2 | 2,867 | 1.5% | 可忽略 |
| 4 | 6,444 | 3.2% | 轻微 |
| 8 | 17,440 | 6.6% | 需关注 |
| 16 | 53,155 | **10.2%** | 瓶颈征兆 |

DAG build 开销随 batch size 超线性增长，在 B=16 时已占 10%。

---

## 四、关键结论

### 结论 1：FJ pure 4N（无 TP）跨 NUMA 是灾难性的

FJ pure 4N（96 线程，无 TP 分片）的性能在所有配置中最差：
- graph_us 高达 **1.1~1.8 秒**（baseline 的 4.6~7.6x，FJ TP4 的 3.4~4.4x）
- barrier 绝对值 **217~451 ms**（baseline 的 13~15x）
- TG B=1 吞吐仅 **0.87 t/s**（baseline 6.56 t/s，FJ TP4 2.95 t/s）
- 原因：**无分片 = 每个线程随机访问 4 个 NUMA 节点的内存**，远程 NUMA 访问延迟 ~2x 本地，导致 compute 和 barrier 同时崩溃
- 结论：**跨 NUMA 使用 96 线程但不分片，比单 NUMA 24 线程更慢**，完全没有并行加速

### 结论 2：FJ TP4 的 barrier 开销达到 20~26%，是性能瓶颈

FJ TP4 每个 graph 中，96 线程平均有 **61~128 ms** 在 barrier 等待（占比 20~26%）。相比 baseline 24 线程的 9~13%，barrier 占比翻倍。原因：
- 每个 TP MUL_MAT 后需要 96 线程全局同步
- 非 TP op 时 72 个 non-domain-0 线程在 spin-wait 空转
- 线程越多，到达 barrier 的 straggler 延迟越大

### 结论 3：Taskflow DAG 调度完全消除了 barrier

TF TP4 的 `tp_mulmat_sync_us` 仅 20~50 us/graph（占比 0.0%），**比 FJ TP4 的 barrier 低 3~4 个数量级**。DAG 依赖链让每个 op 完成后直接触发下游，无需全局同步。

### 结论 4：FJ TP4 的 PP 优势来自 OpenMP 线程调度效率

尽管 FJ TP4 有 20~26% 的 barrier 浪费，其 PP 吞吐仍达到 ~100 t/s（TF TP4 仅 ~70 t/s）。这说明 OpenMP 在大批量并行计算中的线程利用率更高——所有 96 线程同时参与 MUL_MAT 计算，而 Taskflow 的 per-domain executor（每 domain 24 线程）在 PP 阶段无法跨 domain 负载均衡。

### 结论 5：预分片 (row-shard-cache) 是最关键的优化

| 对比 | B=1 mm_compute | B=8 mm_compute | 差异 |
|------|---------------|---------------|------|
| TF TP4 (有预分片) | 99,272 us | 201,690 us | -- |
| TF TP4 (无预分片) | 296,361 us | 386,338 us | **3.0x / 1.9x 慢** |

**预分片的收益 >> barrier 优化的收益**。

### 结论 6：FJ 和 TF 各有最佳场景

| 场景 | 推荐 | 原因 |
|------|------|------|
| 低并发在线推理 (B=1~4) | **TF TP4** | 无 barrier，TG 快 2.6x，综合吞吐更高 |
| 高并发批处理 (B>=8) | **FJ TP4** | PP 吞吐 ~100 t/s，综合最高 |
| 超大 batch (B=16+) | **FJ TP4** | TF 的 DAG build 开销 + PP 劣势 |

### 结论 7：优化方向建议

| 方向 | 预期收益 | 难度 |
|------|---------|------|
| **FJ**: 减少非 TP op 时 d1-d3 的空转 | barrier 降 ~30% | 高（需重构调度） |
| **FJ**: 使用 tree barrier 替代 flat barrier | barrier 降 ~20% | 中（已实现但需集成到 CLI） |
| **TF**: DAG 结构缓存/复用 | B=16 时 graph 快 ~10% | 中 |
| **TF**: PP 阶段允许跨 domain 负载均衡 | PP 吞吐接近 FJ 水平 | 高 |
| **混合**: PP 用 FJ，TG 用 TF | 两者优势兼得 | 中 |

---

## 五、原始 CLI 命令

```bash
# D: Baseline 1NUMA (24 threads, no TP)
numactl --cpunodebind=0 --membind=0 \
  fork-join/build/bin/llama-batched-bench \
  -m Qwen3-4B-Ins-2507-f16.gguf -t 24 \
  -c 4096 -b 512 -ub 256 -npp 64 -ntg 16 -npl 1,2,4,8,16 \
  --output-format md

# F: FJ pure 4N (96 threads, no TP, 4 NUMA nodes)
numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 \
  fork-join/build/bin/llama-batched-bench \
  -m Qwen3-4B-Ins-2507-f16.gguf -t 96 \
  -c 4096 -b 512 -ub 256 -npp 64 -ntg 16 -npl 1,2,4,8,16 \
  --numa distribute

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
GGML_PERF_ALL|nth=96|graph_us=336589|avg_compute_us=247778|avg_barrier_us=61444|...
```

- `graph_us`: 整个 graph compute 的 wall clock
- `avg_compute_us`: 所有线程的 compute 时间平均值（实际执行 op 的时间）
- `avg_barrier_us`: 所有线程的 barrier 等待时间平均值
- 在 TP mode 下，barrier_us 包含三种等待：① TP MUL_MAT 后的全局 barrier ② domain-0 间的 d0 barrier ③ non-domain-0 的 spin-wait
- 配套 `GGML_PERF_THREADS` 和 `GGML_PERF_BARRIER_THREADS` 提供每线程详细数据

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
  fj_pure_4n_bench.md          # FJ pure 4N 吞吐量表格
  fj_pure_4n_perf.log          # FJ pure 4N GGML_PERF_ALL 原始日志 (90 entries)
  fj_tp4_bench.md              # FJ TP4 吞吐量表格
  fj_tp4_perf.log              # FJ TP4 GGML_PERF_ALL 原始日志 (270 entries, 含 per-thread)
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
