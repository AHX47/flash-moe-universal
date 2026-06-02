
# I/O and GPU Exploration: Lessons from Running a 397B Model from SSD

## The Problem

We stream a 397‑billion‑parameter Mixture‑of‑Experts model from NVMe SSD on a MacBook Pro with 48 GB RAM.  
Expert weights total **120 GB at 2‑bit** (209 GB at 4‑bit). Only 6 GB fits in memory.  
Each generated token reads ~600 MB of expert data – 4 experts × 3.9 MB × 60 layers.

**The core question:** Where does the time actually go, and what can we do about it?

---

## Part 1: The GPU Story

### What the Profiler Showed

Metal GPU trace of the expert forward pass (the most compute‑intensive per‑token operation):

| Metric | Value |
|--------|-------|
| Total GPU compute time for 20 expert matvecs | **747 µs** (37 µs each) |
| Wall clock time | **31.5 ms** (1.58 ms each) |
| GPU utilisation | **2.4%** |

The GPU finishes its math in microseconds, then sits idle waiting for data. The bottleneck is **data delivery**, not compute.

### Where GPU Cycles Go

Instruction cost breakdown from Metal performance counters:

| Category | % of GPU Time | What it is |
|----------|:---:|-------------|
| Math (FMA/MUL/ADD) | 63.9% | Dequant + multiply‑accumulate |
| Conversion | 25.0% | `bf16_to_f32()` for scale/bias lookup |
| Data movement | 3.3% | Moving data between register files |
| Bit manipulation | 19.0% | Extracting 2‑bit values from uint32 |

**Observations:**
- **25% of GPU time** is type conversion (bfloat16 → float32). Storing scales/biases as float32 would double storage (negligible) but eliminate a quarter of GPU work.
- **19%** is bit manipulation (extracting 2‑bit nibbles). A lookup table or wider SIMD could reduce this.

**But even eliminating all of this overhead** saves only 44% of 37 µs = 16 µs per matvec.  
At 720 matvecs per token (4 experts × 3 projections × 60 layers), that’s 11.5 ms/token – not nothing, but the I/O bottleneck (90 ms/token) is far larger.

### Cache Behaviour

| Metric | Value |
|--------|-------|
| L1 cache read hit rate | **93.4%** – `x_shared[4096]` works perfectly |
| L1 cache write hit rate | **100%** |
| Last level cache bandwidth | **~418 GB/s** – nearly saturating unified memory |

The GPU cache hierarchy works well. The 6.6% L1 read miss rate is expert weight data that doesn’t fit in L1 (3.9 MB per expert, L1 ~192 KB per core). Misses go to shared L2 / memory fabric at near‑theoretical bandwidth.

### GPU Cluster Affinity Experiment

Apple M3 Max has 40 GPU cores organised into clusters, each with its own L2 cache (~4 MB). 2‑bit experts are 3.9 MB – almost exactly one cluster’s L2 capacity.

**Hypothesis:** Encode all 4 operations for one expert (gate → up → SwiGLU → down) into a single Metal command encoder. The GPU scheduler would keep that work on one cluster, keeping expert weights hot in that cluster’s L2.

**Result:** ❌ **2% slower** – the fused single‑encoder approach reduced parallelism. Metal’s scheduler could no longer overlap work across experts. The existing 2‑encoder‑per‑expert approach (gate+up together, SwiGLU+down together) allows interleaving across clusters, giving better throughput than L2 locality.

**Lesson:** GPU schedulers are smarter than manual NUMA pinning. Don’t fight the hardware scheduler without profiling data.

### What Doesn’t Matter on the GPU

| Attempt | Result |
|---------|--------|
| Superpages (2 MB pages) | ❌ Not available on Apple Silicon ARM64 (fixed 16 KB pages) |
| `commandBufferWithUnretainedReferences` | 🔄 Zero measurable difference |
| Batching all gate+up into one encoder vs separate encoders | 🔄 Zero difference |

---

## Part 2: The I/O Story

### Raw SSD Performance for Expert Reads

| Access Pattern | Throughput | Latency (4 experts) |
|----------------|:---:|:---:|
| Sequential, warm page cache | 32.1 GB/s | 0.49 ms |
| Parallel 4T, warm cache | 29.2 GB/s | 0.97 ms |
| Parallel 4T, cold (`F_NOCACHE`) | 5.5 GB/s | 2.84 ms |
| Sequential, cold | 4.5 GB/s | 3.46 ms |
| `mmap` + `memcpy`, cold | 0.12 GB/s | varies |

The gap between warm (32 GB/s) and cold (5.5 GB/s) is the entire optimisation story. Everything we tried was about moving more data from cold to warm.

### The `mmap` Disaster

**What:** Replace `pread()` with `mmap()` + `memcpy()` for zero‑syscall access to cached data.  
**Result:** 0.56 tok/s – **5× slower** than `pread`.

**Why:** Each 3.9 MB expert spans 240 × 16 KB pages. For uncached data, `mmap` triggers **240 individual page faults**, each requiring a kernel trap → I/O request → page table update. A single `pread()` issues **one** large NVMe command.

**Lesson:** `mmap()` is for random access to already‑cached data. For bulk reads of potentially uncached data, `pread()` is dramatically better – it lets the kernel optimise the I/O pattern.

### The Custom Cache Trap

| Cache Type | Entries | Memory | Hit Rate | tok/s | Verdict |
|------------|:---:|:---:|:---:|:---:|---------|
| None (`pread` only) | 0 | 0 | 0% | 2.86 | Baseline |
| Metal LRU (500) | 500 | 3.5 GB | 35% | 3.14 | Small win |
| Metal LRU (1000) | 1000 | 7.1 GB | 44% | 2.24 | ❌ Worse |
| Metal LRU (2500) | 2500 | 9.8 GB | 55% | 2.24 | ❌ Worse |
| Metal LRU (3000) | 3000 | 21 GB | 55% | 1.99 | ❌ Much worse |
| Malloc zero‑copy (2581) | 2581 | 18 GB | 52% | 2.10 | ❌ Worse |
| **No cache, trust OS** | **0** | **0** | OS‑managed | **5.74** | ✅ **Best** |

**The breakthrough:** Deleting the entire custom cache system gave a **38% speedup** over our best custom implementation.

**Why custom caches hurt:**
1. **Metal buffer caches wire memory** – pinned physical RAM. 9.8 GB wired left only ~25 GB for OS page cache instead of ~35 GB.
2. **OS page cache is smarter** – macOS uses CLOCK‑Pro (adaptive, balances recency and frequency). Our LRU was strictly recency‑based.
3. **Zero lookup overhead** – OS cache hits are normal memory accesses through the MMU. Our cache required hash tables, pointer chasing, and LRU bookkeeping.
4. **Memory pressure compounds** – `vm_stat` showed 60,000‑130,000 decompressions/second with Metal cache active. Without it: near zero.

**Database analogy:** PostgreSQL recommends `shared_buffers` at 25% of RAM, letting the OS cache handle the rest. We were doing the equivalent of 60% – squeezing out the OS cache that handles long‑tail access patterns better than any application cache.

### Kernel Hint Experiments

| Hint | Purpose | Result | Why |
|------|---------|:---:|-----|
| `F_NOCACHE` | Bypass page cache | +3% (2‑bit) | Avoids thrashing, but prevents warm hits |
| `F_RDAHEAD` | Enable readahead | 0% | Kernel already does it |
| `F_RDADVISE` (immediate) | Pre‑hint reads | -8% | NVMe command contention |
| `F_RDADVISE` (lead time) | Pre‑hint from previous token | -4% | 65‑80% wrong predictions |
| `MADV_RANDOM` | Disable readahead | ❌ Harmful | Fragments reads |
| `MADV_SEQUENTIAL` | Large readahead | 0% | Fragmentation is physical page layout |
| `MADV_WILLNEED` | Pre‑populate cache | 0% | Only helps first access |
| **No hint (default)** | Let kernel decide | ✅ **Best** | Kernel already well‑tuned |

**The pattern:** Every hint was neutral or harmful. macOS default I/O behaviour is already optimal for Apple hardware.

### Fragmentation Discovery

`fs_usage` profiling revealed:

```
pread calls:     45,414
RdData ops:     260,845
Reads per pread: 5.7×
```

Each 3.9 MB `pread` is broken into ~5.7 separate NVMe commands (mostly 512 KB). Reason: page cache stores data in scattered 16 KB virtual pages that map to non‑contiguous physical pages. Kernel cannot coalesce them.

**Block size distribution:**
- 76,549 × 512 KB
- 38,441 × 8 KB
- 20,064 × 16 KB
- 17,647 × 12 KB
- 14,651 × 256 KB

Fragmentation adds ~46 µs kernel overhead per `pread`. 240 `pread` calls/token × 46 µs = **11 ms/token**. Inherent to virtual memory – cannot be fixed from userspace.

### Buffer Alignment Matters

| Buffer Alignment | Avg Latency | Throughput |
|------------------|:---:|:---:|
| 2 MB‑aligned (`posix_memalign`) | 234 µs | 16.8 GB/s |
| 16 KB‑aligned (default Metal) | 836 µs | 4.7 GB/s |

**3.6× faster with 2 MB alignment** for page‑cache‑resident data. DMA controller can do larger burst transfers. In the full pipeline, improvement was ~5% (most reads hit SSD where NAND latency dominates). Still a free optimisation.

### Tiered I/O Experiment

**Hypothesis:** Two file descriptors per layer – one with `F_NOCACHE` for first‑time reads, one without for repeats. Track “seen” experts with a 3.8 KB bitset.

**Result:** Marginally better tok/s, but `vm_stat` showed identical memory pressure. The pressure came from Metal buffers and model weights, not page cache behaviour. Added complexity without meaningful benefit.

### What Actually Worked for I/O

1. **2‑bit expert quantisation** – 44% smaller files. Reduced `expert_io` from 2.6 ms to 1.5 ms per layer. The single biggest improvement.
2. **Trust the OS page cache** – Delete custom caches. 38% speedup.
3. **2 MB‑aligned DMA buffers** – 5% improvement on `expert_io`. Free.
4. **Parallel `pread` (4 threads)** – 9.2× speedup over sequential (superlinear due to NVMe command queuing).
5. **No kernel hints** – Default behaviour is optimal.

---

## Part 3: The Bigger Picture

### The SSD Bandwidth Wall

At 2‑bit precision, theoretical I/O floor:

```
60 layers × ~2.6 cache misses × 3.9 MB = 608 MB per token
608 MB ÷ 5.5 GB/s (random read) = 110 ms → 9.1 tok/s (I/O limited)
```

Our measured performance: 5.5 tok/s (182 ms/token), split roughly **50/50 between I/O (90 ms) and compute (90 ms)**. We’re at **82% of I/O‑limited theoretical maximum**.

The remaining 18% gap is page cache fragmentation overhead (5.7 ops/`pread`) and kernel per‑read overhead (46 µs/`pread`) – architectural limitations of macOS virtual memory.

### Systems Thinking Beats Micro‑Optimisation

The single most impactful change was **deleting code** – removing the 9.8 GB Metal buffer cache. It wasn’t that the cache was poorly implemented; it was that its existence created system‑level effects (memory pressure, compressor thrashing, reduced page cache) that outweighed its direct benefits.

**Classic systems engineering lesson:** Optimising one component in isolation can degrade the whole system.
- GPU profiling showed compute isn’t the bottleneck.
- I/O profiling showed the kernel is already doing a good job.
- `vm_stat` showed our “optimisation” was causing the real problem.

### What the Database World Already Knew

Dan Woods’ key insight: **treat model weights like a database.** Databases solved the problem of datasets larger than memory decades ago:

- **Don’t build your own buffer pool.** PostgreSQL learned this – `shared_buffers` at 25% of RAM, not 100%. OS buffer cache handles the long tail better.
- **Respect the hardware cache hierarchy.** Don’t bypass caches (`F_NOCACHE`) without measured evidence of thrashing.
- **Profile before optimising.** `fs_usage` and Metal traces told us exactly where time goes.
- **Alignment matters for DMA.** Database systems align I/O buffers to page boundaries. 2 MB alignment gave 3.6× better DMA throughput.

---

## Remaining Frontiers

| Frontier | Status |
|----------|--------|
| Batch prefill | Process multiple prompt tokens simultaneously. Sequential GatedDeltaNet limits parallelism, but projection matmuls and expert I/O can be batched. |
| C tokenizer | ✅ **Done** – eliminated 3.5 s Python overhead (4 s → 180 ms startup) |
| Page cache fragmentation | 5.7 disk ops per `pread` is a kernel limitation. Userspace mitigation: use `mincore()` to detect cached pages and `memcpy()` from `mmap` for hits, falling back to `pread` for misses. |
| Expert file layout optimisation | Co‑locating frequently co‑accessed experts could reduce distinct NVMe commands per token. Requires offline analysis of routing patterns. |

---

**End of document**
