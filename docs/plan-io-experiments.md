
# I/O Optimization Experiments – Plan & Analysis

## Baseline Reference

| Metric | Value |
|--------|-------|
| Per‑layer total | 4.28 ms |
| Expert I/O (`expert_io`) | **2.41 ms (56%)** |
| Expert read pattern | 4 × 7 MB parallel `pread` from 3.4 GB file |
| Measured I/O times | 5.8 ms (cold parallel), 1.0 ms (warm parallel), 2.4 ms mixed (71% cache hit) |
| Theoretical floor | 28 MB / 17.5 GB/s = **1.6 ms** |
| Overhead gap | 0.8 ms per layer (kernel VFS + page cache + NVMe scheduling) |

---

## Experiment 1: `dispatch_io` vs `pread`

### Hypothesis

`dispatch_io` creates a kernel‑side I/O channel that can reorder NVMe commands by LBA, potentially reducing overhead compared to 4 separate `pread` calls.

### Isolated Test

Read 4 experts (7 MB each) from a layer file using:

- **(A)** 4 × `pread` on 4 pthreads (baseline)
- **(B)** 4 × `dispatch_io_read` on a `DISPATCH_IO_RANDOM` channel

Both with `F_NOCACHE` to force SSD reads. Warm cache. 50 iterations each.

**Metrics:** Wall time, throughput (GB/s logical).

### What `dispatch_io` Does Differently

- Kernel‑side I/O channel with optimised scheduling
- Kernel sees all reads as part of one channel → can reorder by LBA
- Automatic cleanup (no thread join overhead)
- May use a VFS path optimised for random access

### Contention Analysis

- `dispatch_io` is async with completion blocks on GCD queues.
- Completion block runs on a GCD thread (same as `async_pread`).
- Memory path unchanged: SSD → DMA → DRAM → Metal shared buffer.
- **No new memory contention.**

**Risks:**
- GCD overhead may exceed VFS savings.
- Completion blocks may have higher latency than pthread wakeup.

### Expected Impact

| Case | Impact on `expert_io` |
|------|----------------------|
| Best | 10‑20% reduction (eliminates per‑syscall VFS overhead) |
| Worst | Neutral or slight regression |

**Pipeline impact:** Drop‑in replacement for `pread`, no GPU interaction.

---

## Experiment 2: GPU Private Buffer Compression

### Hypothesis

`StorageModePrivate` buffers allow GPU hardware memory compression, doubling effective bandwidth for compressible data (4‑bit quantised weights have 2.4‑3.7 bits entropy).

### Isolated Test

- **(A)** Baseline: matvec reading from `StorageModeShared` buffer
- **(B)** GPU blit `shared→private`, then matvec from `StorageModePrivate`

Same expert weights, same kernel, same dimensions. Measure blit time, matvec time, total. 100 iterations.

### How GPU Compression Works

- `StorageModePrivate` buffers live in GPU‑managed memory.
- Memory controller applies **lossless compression** (transparent to shaders).
- For compressible data: effective bandwidth doubles (read 64 B, decompress to 128 B).
- 4‑bit quantised weights have 2.4‑3.7 bits entropy → highly compressible.
- Shader code unchanged – compression is hardware‑transparent.

### Contention Analysis

- Blit (`shared→private`) runs on GPU command queue **before** matvec dispatches.
- Timeline: `[pread→shared buf]` → `[GPU blit ~0.02ms]` → `[GPU matvec from private]`
- Blit adds ~0.02 ms per expert to `CMD3`.
- Matvec may be 30‑50% faster from doubled bandwidth.
- **Key constraint:** Blit and matvec are on the **same GPU queue** (serial). Net benefit depends on whether bandwidth gain exceeds blit cost.
- **Memory pressure:** 4 × 7 MB = 28 MB private memory per layer. Must reuse/recycle buffers each layer.

### Expected Impact

| Case | Impact on `cmd1_wait` |
|------|----------------------|
| Best | 15‑30% reduction |
| Worst | Slight regression (blit cost > compression gain) |

**Pipeline impact:** GPU‑only, no SSD interaction.

---

## Experiment 3: Expert File Clustering by Co‑occurrence

### Hypothesis

NVMe reads large sequential blocks faster than scattered reads. Reorder experts so frequently co‑occurring experts are adjacent in the file.

### Isolated Test

1. Run 500 tokens with `--freq` to collect per‑layer expert co‑occurrence matrix.
2. For each layer: cluster 512 experts so frequently co‑occurring experts are adjacent.
3. Repack layer file with new ordering + save permutation map.
4. Compare 4‑expert parallel `pread` with original vs clustered ordering.
5. Use same expert indices (mapped through permutation), `F_NOCACHE`, 50 iterations.

### What Clustering Does

- NVMe reads in pages (4‑16 KB). Reading expert 37 (7 MB at offset 262 MB) reads pages 262.0‑269.0 MB. Expert 38 is at 269‑276 MB – adjacent.
- If routing selects `{37, 42, 100, 205}`, offsets are `{262, 297, 708, 1451}` MB – widely scattered.
- After clustering: `{37, 42, 100, 205}` might become physical positions `{0, 1, 2, 3}` – a **28 MB sequential read** instead of 4 scattered reads.
- Sequential 28 MB at 17.5 GB/s = 1.6 ms vs scattered 4×7 MB at ~5.8 ms.

### Contention Analysis

- Changes **file layout only** – inference code reads the same way.
- Expert indices mapped through permutation table (one array lookup, ~0 ns).
- **No contention** – purely changes which bytes are at which file offsets.
- **Risk:** Co‑occurrence patterns may be prompt‑dependent. Mitigate with diverse profiling prompts.

### Expected Impact

| Case | Impact on cold read `expert_io` |
|------|-------------------------------|
| Best | 30‑50% reduction (scattered → near‑sequential) |
| Worst | Neutral (flat co‑occurrence) |

**Pipeline impact:** Pure I/O improvement, no GPU/CPU interaction.

---

## Experiment 4: LZ4 DRAM Expert Cache (Low Priority)

### Hypothesis

Store compressed experts in userspace DRAM as a second‑level cache between OS page cache and SSD.

### Isolated Test

- Allocate 4 GB of `malloc` memory.
- After each expert `pread`, LZ4‑compress and store (hash by layer+expert_id).
- On subsequent reads: check cache first. Hit = LZ4 decompress from DRAM.
- Measure: cache hit rate over 200 tokens, avg expert read time (hit vs miss).

### Cache Maths

| Metric | Calculation |
|--------|-------------|
| 4 GB LZ4 cache holds | ~730 compressed experts |
| OS page cache (reduced from 35 GB → 31 GB) | ~4430 raw expert slots |
| Total accessible experts | 4430 + 730 = 5160 |
| Gain over baseline (5000) | **only 3%** |
| 8 GB LZ4 cache | 3857 + 1455 = 5312 – still marginal |

**The math doesn’t work** unless the LZ4 cache has a **much** higher hit rate than the page cache (e.g., via smarter eviction policy).

### Contention Analysis

- Cache lookup: hash table check (~0.001 ms) – negligible.
- Cache hit: LZ4 decompress on I/O worker thread (CPU busy for 0.17 ms, overlaps with other threads).
- Cache miss: normal `pread` path.
- **Memory contention:** 4 GB of malloc’d DRAM reduces OS page cache by 4 GB. The small gain (3%) likely does not justify the complexity.

### Expected Impact

| Case | Impact on `expert_io` |
|------|----------------------|
| Best | 5‑10% reduction |
| Worst | Negative (reduced page cache hurts more) |

**Verdict:** ❌ Probably not worth implementing.

---

## Experiment 5: `aio_read` Batching

### Hypothesis

Kernel‑level async I/O sees all 4 requests at once and can batch NVMe commands, eliminating per‑thread overhead.

### Isolated Test

- **(A)** 4 × `pread` on 4 pthreads (baseline)
- **(B)** 4 × `aio_read`, then `aio_suspend` to wait for all

Both `F_NOCACHE`, warm cache, 50 iterations.

### What `aio_read` Does Differently

- Submits I/O requests to kernel without blocking the calling thread.
- Kernel sees all 4 requests at once and can batch NVMe commands.
- `aio_suspend` blocks until all 4 complete (single wait vs 4 thread joins).
- Eliminates per‑thread overhead (no `pthread_create`/join or `dispatch_group`).

### Contention Analysis

- Uses kernel‑level async I/O (not userspace threads).
- Kernel I/O scheduler has full visibility into all pending reads.
- **No new contention** – same SSD path, potentially better NVMe scheduling.
- **Risks:**
  - macOS `aio` implementation may be less optimised than GCD.
  - Completion notification (`SIGEV_THREAD` or `SIGEV_SIGNAL`) has latency.

### Expected Impact

| Case | Impact on `expert_io` |
|------|----------------------|
| Best | 5‑15% reduction |
| Worst | Neutral or regression |

**Pipeline impact:** Drop‑in replacement for `pread`.

---

## Execution Priority

| # | Experiment | Expected Impact | Effort | Risk | Priority |
|---|------------|----------------|--------|------|----------|
| 3 | Expert clustering | **30‑50%** cold I/O | Medium | Low | **1st** |
| 1 | `dispatch_io` | 10‑20% I/O | Low | Low | **2nd** |
| 2 | GPU private compression | 15‑30% GPU | Medium | Medium | **3rd** |
| 5 | `aio_read` | 5‑15% I/O | Low | Low | **4th** |
| 4 | LZ4 DRAM cache | 5‑10% I/O | High | High | **Skip** |

---

## Compound Effects

Test each in isolation first – improvements that look good alone can cancel each other (e.g., `F_RDADVISE` gave `expert_io -31%` but `cmd2_wait +73%` = net 0%).

After isolated tests:

| Combination | Expected Outcome |
|-------------|------------------|
| 1 + 3 (`dispatch_io` + clustering) | Should compound – clustering reduces scatter, `dispatch_io` optimises remaining scatter |
| 2 (GPU compression) | GPU‑only, orthogonal to I/O experiments – should compound cleanly |
| 5 (alternative to 1) | Test both, pick winner |

---

## Conclusion

**Expert clustering (Experiment 3)** has the highest potential impact (30‑50% reduction in cold I/O) with low risk and no pipeline contention. It should be the first experiment to implement.

If clustering alone doesn’t close the gap, `dispatch_io` (Experiment 1) and GPU private compression (Experiment 2) are promising secondary optimisations.

LZ4 DRAM cache is mathematically unattractive and not worth the implementation effort.

