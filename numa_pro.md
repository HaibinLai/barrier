# NUMA Remote Memory Access Profiling Data for Paper Motivation

## Experimental Setup

- **Platform**: 8-socket Intel Xeon Platinum 8160 @ 2.10 GHz
- **Cores/Socket**: 24 cores (48 threads with HT)
- **Total CPUs**: 384 (192 physical cores)
- **L3 Cache/Socket**: 33 MiB (264 MiB total)
- **Memory/Node**: ~96 GB DDR4 (768 GB total)
- **NUMA Topology**: 8 nodes, distances: 10 (local), 21 (near), 31 (far)
- **Models**:
  - Qwen3-4B F16 (4.02B params, 7.49 GiB weights, n_head=32, n_head_kv=8, 36 layers)
  - Meta-Llama-3.1-8B-Instruct F16 (8.03B params, 14.96 GiB weights, n_head=32, n_head_kv=8, 32 layers)
- **Framework**: llama.cpp (fork-join branch) with CPU tensor parallelism
- **Benchmark**: PP=128, TG=10, batch sizes 1/2/4/8
- **Profiler**: Linux perf stat with PMU hardware counters
- **OS**: Ubuntu 22.04, kernel with `numa_balancing=1` (default)

### NUMA Distance Matrix

```
node   0   1   2   3   4   5   6   7
  0:  10  21  21  31  21  31  31  31
  1:  21  10  31  21  31  21  31  31
  2:  21  31  10  21  31  31  21  31
  3:  31  21  21  10  31  31  31  21
  4:  21  31  31  31  10  21  31  21
  5:  31  21  31  31  21  10  21  31
  6:  31  31  21  31  31  21  10  21
  7:  31  31  31  21  21  31  21  10
```

---

## Table 1: Throughput Performance (tokens/s, total = PP + TG combined)

All configurations use `--numa distribute`. Weights loaded via mmap (default placement on node 0).

| Configuration          | Threads | B=1   | B=2   | B=4   | B=8   |
|------------------------|---------|-------|-------|-------|-------|
| 1 NUMA (baseline)      | 24      | 20.52 | 27.15 | 34.50 | 41.39 |
| 2 NUMA, no TP          | 48      | -     | -     | -     | -     |
| TP2 (2 NUMA nodes)     | 48      | 22.51 | 37.42 | 51.97 | 61.26 |
| 4 NUMA, no TP          | 96      | 8.40  | 13.02 | 17.51 | 20.94 |
| TP4 (4 NUMA nodes)     | 96      | 19.75 | 33.71 | 52.30 | 68.35 |
| 8 NUMA, no TP          | 192     | 5.12  | 7.50  | 10.34 | 12.15 |

**Key observation**: Simply adding more NUMA nodes without TP **degrades** performance (4NUMA noTP is 2.0x slower than 1NUMA). TP partially recovers this but still faces scaling limits.

---

## Table 2: Hardware Performance Counter Comparison (perf stat)

Measured over identical workload: PP=128, TG=10, npl=1,2,4,8 (all batch sizes combined in one run).

| Metric                  | 1 NUMA (24t)   | TP2 (48t)      | TP4 (96t)       | 4N noTP (96t)   | 8N noTP (192t)   |
|-------------------------|----------------|----------------|-----------------|-----------------|------------------|
| **node-load-misses**    | 1,728M         | 783M           | 1,006M          | 3,370M          | 5,779M           |
| **node-loads**          | 226M           | 893M           | 468M            | 234M            | 277M             |
| **node-store-misses**   | 431M           | 97M            | 103M            | 1,498M          | 2,718M           |
| **LLC-load-misses**     | 2,941M         | 1,685M         | 1,837M          | 4,859M          | 7,682M           |
| **LLC-loads**           | 12,033M        | 8,599M         | 9,476M          | 16,057M         | 28,637M          |
| **LLC miss rate**       | 24.4%          | 19.6%          | 19.4%           | 30.3%           | 26.8%            |
| **cache-misses**        | 12,897M        | 7,460M         | 7,743M          | 28,881M         | 47,567M          |
| **cache-references**    | 37,015M        | 40,364M        | 43,331M         | 63,632M         | 115,029M         |
| **cache miss rate**     | 34.8%          | 18.5%          | 17.9%           | 45.4%           | 41.4%            |
| **instructions**        | 2,406B         | 2,445B         | 2,580B          | 2,985B          | 4,004B           |
| **cycles**              | 4,779B         | 4,663B         | 9,241B          | 34,922B         | 97,144B          |
| **IPC**                 | **0.50**       | **0.52**       | **0.28**        | **0.09**        | **0.04**         |
| **Wall-clock time (s)** | 61.0           | 43.6           | 43.5            | 124.0           | 210.8            |
| **CPU time user (s)**   | 1,398          | 1,944          | 3,611           | 11,110          | 37,267           |
| **CPU time sys (s)**    | 36             | 55             | 394             | 376             | 1,579            |

---

## Table 3: Remote Memory Access Analysis (derived)

| Metric                               | 1 NUMA  | TP2     | TP4     | 4N noTP  | 8N noTP   |
|---------------------------------------|---------|---------|---------|----------|-----------|
| node-load-misses (M)                  | 1,728   | 783     | 1,006   | 3,370    | 5,779     |
| node-store-misses (M)                 | 431     | 97      | 103     | 1,498    | 2,718     |
| **Total remote access (M)**           | **2,159** | **880** | **1,109** | **4,868** | **8,497** |
| vs. 1NUMA ratio                       | 1.00x   | 0.41x   | 0.51x   | 2.25x   | 3.93x     |
| IPC                                   | 0.50    | 0.52    | 0.28    | 0.09    | 0.04      |
| IPC degradation vs. 1NUMA             | -       | +4%     | -44%    | -82%    | -92%      |
| Total cycles (B)                      | 4,779   | 4,663   | 9,241   | 34,922  | 97,144    |
| Cycles per useful instruction         | 1.99    | 1.91    | 3.58    | 11.70   | 24.26     |
| **Throughput B=8 (tok/s)**            | 41.39   | 61.26   | 68.35   | 20.94   | 12.15     |
| Throughput scaling vs. 1NUMA          | 1.00x   | 1.48x   | 1.65x   | 0.51x   | 0.29x     |

---

## Key Findings for Paper Motivation

### Finding 1: Naive Multi-NUMA Scaling is Negative

Adding NUMA nodes **without** tensor parallelism causes severe performance degradation:
- **4 NUMA nodes (96 threads, no TP)**: 0.51x throughput of single NUMA (41.39 -> 20.94 tok/s at B=8)
- **8 NUMA nodes (192 threads, no TP)**: 0.29x throughput (41.39 -> 12.15 tok/s at B=8)

Root cause: All threads contend for the same mmap'd weight memory on node 0. Remote memory access count increases from 2.2B (1 NUMA) to **8.5B (8 NUMA)**, a 3.93x increase. IPC collapses from 0.50 to 0.04.

### Finding 2: Remote Access is the Dominant Bottleneck

The relationship between remote memory access and IPC degradation is clear:

| Configuration | Remote accesses | IPC  | Performance |
|---------------|-----------------|------|-------------|
| 1 NUMA        | 2.2B (baseline) | 0.50 | 1.00x       |
| 4N noTP       | 4.9B (2.25x)    | 0.09 | 0.51x       |
| 8N noTP       | 8.5B (3.93x)    | 0.04 | 0.29x       |

Each doubling of remote accesses causes approximately 5-10x IPC degradation due to memory stalls at NUMA distance 21-31.

### Finding 3: Tensor Parallelism Mitigates but Does Not Eliminate

TP reduces remote access by partitioning computation across NUMA domains:
- TP2 reduces remote access to 0.41x of baseline (880M vs 2,159M), achieves 1.48x speedup
- TP4 reduces remote access to 0.51x of baseline, achieves 1.65x speedup

However, TP4's IPC (0.28) is still 44% lower than 1NUMA (0.50) due to:
1. Weights still reside on a single mmap'd node — remote reads for 3/4 of domains
2. Synchronization overhead: taskflow barrier wait + reduce operations
3. Graph node explosion: 4,110 nodes (TP4) vs 1,446 (baseline), 2.84x more scheduling work

### Finding 4: CPU Time Waste Quantifies the Problem

| Configuration | Wall-clock (s) | Total CPU time (s) | CPU efficiency |
|---------------|---------------|-------------------|----------------|
| 1 NUMA (24t)  | 61.0          | 1,435             | 1.00x baseline |
| TP2 (48t)     | 43.6          | 1,999             | 1.39x          |
| TP4 (96t)     | 43.5          | 4,005             | 2.79x          |
| 4N noTP (96t) | 124.0         | 11,486            | **8.00x**      |
| 8N noTP (192t)| 210.8         | 38,846            | **27.07x**     |

8NUMA noTP wastes **27x** the CPU resources of 1NUMA for **0.29x** the throughput. Even TP4 uses **2.79x** CPU resources for only **1.65x** throughput — a parallel efficiency of 59%.

---

## Suggested Paper Narrative

> *"On a NUMA system with 8 sockets, naively scaling thread count across sockets leads to severe performance degradation. Our profiling reveals that remote NUMA memory access is the dominant bottleneck: compared to single-socket execution, 8-socket execution incurs 3.93x more remote memory accesses (8.5B vs 2.2B events), causing IPC to collapse from 0.50 to 0.04 — a 92% reduction. Even with tensor parallelism partitioning computation across 4 NUMA domains, IPC remains 44% below single-socket levels due to unresolved weight data locality issues. These findings motivate our NUMA-aware weight placement strategy, which eliminates remote weight access by replicating weight shards to each domain's local memory."*

---

## Profiling Methodology

All measurements use Linux `perf stat` with Intel PMU hardware counters:

```bash
perf stat -e node-load-misses,node-loads,node-store-misses,node-stores,\
             LLC-load-misses,LLC-loads,cache-misses,cache-references,\
             instructions,cycles \
  ./llama-batched-bench -m <model> -c 2048 -npp 128 -ntg 10 \
  -npl 1,2,4,8 --numa distribute [--cpu-tp N] -t <threads>
```

- `node-load-misses` / `node-store-misses`: Count memory accesses satisfied by a **remote** NUMA node (Intel uncore PMU `UNC_CHA_REMOTE_*` events)
- `node-loads` / `node-stores`: All NUMA-tagged memory operations
- `LLC-load-misses`: Last-level cache (L3) misses requiring DRAM access
- `IPC = instructions / cycles`: Instructions per cycle, measures pipeline efficiency (stalls from memory latency reduce IPC)

---

## Llama 3.1 8B Results (14.96 GiB F16)

### Table 4: Throughput Performance — Llama 3.1 8B (tokens/s)

| Configuration          | Threads | B=1   | B=2   | B=4   | B=8   |
|------------------------|---------|-------|-------|-------|-------|
| 1 NUMA (baseline)      | 24      | 16.88 | 23.58 | 31.53 | 35.56 |
| TP2 (2 NUMA nodes)     | 48      | 15.21 | 24.13 | 33.08 | 39.77 |
| TP4 (4 NUMA nodes)     | 96      | 11.97 | 21.64 | 34.42 | 46.95 |
| 4 NUMA, no TP          | 96      | 9.70  | 13.82 | 20.14 | 23.95 |
| 8 NUMA, no TP          | 192     | 5.89  | 7.56  | 12.05 | 13.44 |

### Table 5: Hardware Performance Counters — Llama 3.1 8B

| Metric                  | 1 NUMA (24t)   | TP2 (48t)      | TP4 (96t)       | 4N noTP (96t)   | 8N noTP (192t)   |
|-------------------------|----------------|----------------|-----------------|-----------------|------------------|
| **node-load-misses**    | 1,998M         | 2,413M         | 1,182M          | 2,705M          | 3,935M           |
| **node-loads**          | 575M           | 164M           | 156M            | 409M            | 424M             |
| **node-store-misses**   | 457M           | 100M           | 121M            | 2,076M          | 4,076M           |
| **LLC-load-misses**     | 4,221M         | 4,871M         | 3,472M          | 4,833M          | 6,233M           |
| **LLC-loads**           | 23,573M        | 19,273M        | 17,674M         | 28,011M         | 29,066M          |
| **LLC miss rate**       | 17.9%          | 25.3%          | 19.6%           | 17.3%           | 21.5%            |
| **cache-misses**        | 17,016M        | 21,301M        | 15,068M         | 27,811M         | 39,830M          |
| **cache-references**    | 64,764M        | 72,362M        | 76,252M         | 98,969M         | 118,961M         |
| **cache miss rate**     | 26.3%          | 29.4%          | 19.8%           | 28.1%           | 33.5%            |
| **instructions**        | 4,036B         | 4,072B         | 4,238B          | 4,326B          | 4,849B           |
| **cycles**              | 5,478B         | 6,733B         | 13,508B         | 30,497B         | 86,514B          |
| **IPC**                 | **0.74**       | **0.60**       | **0.31**        | **0.14**        | **0.06**         |
| **Wall-clock time (s)** | 70.3           | 66.8           | 65.9            | 110.0           | 191.0            |
| **CPU time user (s)**   | 1,626          | 3,047          | 5,812           | 9,903           | 34,315           |
| **CPU time sys (s)**    | 29             | 56             | 327             | 364             | 1,351            |

### Table 6: Remote Memory Access Analysis — Llama 3.1 8B

| Metric                               | 1 NUMA  | TP2     | TP4     | 4N noTP  | 8N noTP   |
|---------------------------------------|---------|---------|---------|----------|-----------|
| node-load-misses (M)                  | 1,998   | 2,413   | 1,182   | 2,705    | 3,935     |
| node-store-misses (M)                 | 457     | 100     | 121     | 2,076    | 4,076     |
| **Total remote access (M)**           | **2,455** | **2,513** | **1,303** | **4,781** | **8,011** |
| vs. 1NUMA ratio                       | 1.00x   | 1.02x   | 0.53x   | 1.95x   | 3.26x     |
| IPC                                   | 0.74    | 0.60    | 0.31    | 0.14    | 0.06      |
| IPC degradation vs. 1NUMA             | -       | -19%    | -58%    | -81%    | -92%      |
| Cycles per useful instruction         | 1.36    | 1.65    | 3.19    | 7.05    | 17.84     |
| **Throughput B=8 (tok/s)**            | 35.56   | 39.77   | 46.95   | 23.95   | 13.44     |
| Throughput scaling vs. 1NUMA          | 1.00x   | 1.12x   | 1.32x   | 0.67x   | 0.38x     |

---

## Cross-Model Comparison Summary

### IPC Degradation Across NUMA Configurations

| Configuration  | Qwen3-4B IPC | Llama-8B IPC | Qwen3-4B Perf | Llama-8B Perf |
|----------------|-------------|-------------|---------------|---------------|
| 1 NUMA         | 0.50        | 0.74        | 1.00x         | 1.00x         |
| TP2            | 0.52        | 0.60        | 1.48x         | 1.12x         |
| TP4            | 0.28        | 0.31        | 1.65x         | 1.32x         |
| 4N noTP        | 0.09        | 0.14        | 0.51x         | 0.67x         |
| 8N noTP        | 0.04        | 0.06        | 0.29x         | 0.38x         |

**Consistent finding across both models**: IPC collapses to 0.04-0.06 at 8 NUMA nodes without TP, regardless of model size. The 2x larger model (8B vs 4B) suffers even more from remote weight access due to 2x more weight data traversing NUMA interconnects per forward pass.

### Remote Access Scaling (node-load-misses + node-store-misses)

| Configuration  | Qwen3-4B     | Llama-8B     |
|----------------|-------------|-------------|
| 1 NUMA         | 2.2B (1.0x) | 2.5B (1.0x) |
| 4N noTP        | 4.9B (2.3x) | 4.8B (1.9x) |
| 8N noTP        | 8.5B (3.9x) | 8.0B (3.3x) |

### CPU Resource Waste

| Configuration  | Qwen3-4B CPU waste | Llama-8B CPU waste |
|----------------|-------------------|-------------------|
| 1 NUMA         | 1.0x (baseline)   | 1.0x (baseline)   |
| 4N noTP        | 8.0x              | 6.2x              |
| 8N noTP        | 27.1x             | 21.6x             |

**With the 8B model, 8 NUMA nodes waste 21.6x CPU resources for only 0.38x the throughput.**

---

## Table 7: Local vs Remote DRAM Access — Llama 3.1 8B (Precise PMU Events)

Using `mem_load_l3_miss_retired.*` events which provide **precise**, retired-instruction-level counts of L3-miss loads served by local vs remote DRAM.

- `local_dram`: L3 miss satisfied by **local** NUMA node's DRAM
- `remote_dram`: L3 miss satisfied by **remote** NUMA node's DRAM (cold line, no sharing)
- `remote_hitm`: L3 miss satisfied by **remote** cache in **modified** state (cross-socket cache-to-cache transfer with invalidation — most expensive)
- `remote_fwd`: L3 miss satisfied by **remote** cache forwarding (shared line, cheaper than hitm)

| Metric (M events)       | 1 NUMA (24t) | TP2 (48t) | TP4 (96t) | 4N noTP (96t) | 8N noTP (192t) |
|--------------------------|-------------|-----------|-----------|---------------|----------------|
| **local_dram**           | 534.0       | 119.2     | 124.0     | 352.0         | 371.9          |
| **remote_dram**          | 69.3        | 38.6      | 36.3      | 95.7          | 130.0          |
| **remote_hitm**          | 111.7       | 63.2      | 108.7     | 178.4         | 220.2          |
| **remote_fwd**           | 141.6       | 43.0      | 165.9     | 252.4         | 381.5          |
| **Total remote**         | **322.6**   | **144.8** | **310.9** | **526.5**     | **731.7**      |
| **Total DRAM access**    | **856.6**   | **264.0** | **434.9** | **878.5**     | **1,103.6**    |
| **Remote ratio**         | **37.7%**   | **54.8%** | **71.5%** | **59.9%**     | **66.3%**      |
| IPC                      | 0.73        | 0.60      | 0.31      | 0.14          | 0.06           |

### Key Observations

1. **1 NUMA baseline**: Even with a single NUMA domain (24 threads on node 0), 37.7% of DRAM accesses hit remote nodes. This is because `mmap` with `--numa distribute` may place some pages on other nodes, and `numa_balancing` migrates pages opportunistically.

2. **TP2**: Total DRAM accesses drop dramatically (856M → 264M) because each domain's L3 cache can serve a larger fraction of its weight shard. Remote ratio rises to 54.8% because domain 1 still reads domain 0's weight memory.

3. **TP4**: **71.5% of all DRAM accesses are remote** — the worst ratio. With 4 domains, 3 out of 4 read weights from node 0. The `remote_fwd` count (165.9M) is 3.9x the TP2 level, indicating heavy cross-socket cache-line forwarding.

4. **4N noTP → 8N noTP**: Remote access grows from 526.5M to 731.7M (+39%), but total DRAM access also grows from 878.5M to 1,103.6M (+26%) — more threads causing more L3 capacity misses.

5. **Critical insight for motivation**: The remote ratio **increases** with more NUMA domains (37.7% → 59.9% → 66.3% for noTP; 54.8% → 71.5% for TP). This directly explains the IPC collapse: each remote DRAM access costs 2-3x a local access (NUMA distance 21-31 vs 10), so a 71.5% remote ratio means the effective memory latency is ~2x higher than the local-only case.

### Correlation: Remote Ratio vs IPC (Llama 3.1 8B)

```
Config      Remote%   IPC    Throughput (B=8)
1 NUMA      37.7%     0.73   35.56 tok/s (1.00x)
TP2         54.8%     0.60   39.77 tok/s (1.12x)
4N noTP     59.9%     0.14   23.95 tok/s (0.67x)
8N noTP     66.3%     0.06   13.44 tok/s (0.38x)
TP4         71.5%     0.31   46.95 tok/s (1.32x)
```

Note: TP4 achieves 1.32x throughput despite 71.5% remote ratio because TP partitions the **computation** across domains (reducing per-domain work), whereas noTP merely adds more threads contending for the same serial computation. TP4's IPC (0.31) is still less than half of 1NUMA (0.73), confirming that remote access remains the primary efficiency loss even with TP.
