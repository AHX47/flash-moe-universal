

# 4‑Bit Expert Optimization Experiments

## Context

After discovering that 2‑bit expert quantization broke tool calling (JSON quotes → backslashes), we reverted to 4‑bit experts. This dropped performance from **5.74 tok/s (2‑bit)** to **3.50 tok/s (4‑bit)**.

**Goal:** Recover as much speed as possible while maintaining 4‑bit quality.

**Expert size (4‑bit):** 7,077,888 bytes (6.75 MB)  
**With K=4 active experts per layer and 60 layers:**  
Each token reads 240 experts = **1.68 GB from SSD**.

---

## Baseline Pipeline (4‑bit, K=4, Trust OS page cache)

| Phase | Time | % | Description |
|-------|------|---|-------------|
| `cmd1_wait` | 1.22 ms | 28% | GPU: CMD3(prev) + CMD1 attention projections |
| `cmd2_wait` | 0.55 ms | 13% | GPU: o_proj + norm + routing + shared expert |
| `expert_io` | **2.41 ms** | **56%** | SSD: 4×7 MB parallel pread |
| CPU work | 0.10 ms | 2% | encode + attention + routing + memcpy |

**Per layer:** 4.28 ms → 60 layers = 257 ms/token = **3.90 tok/s**

Page cache hit rate: ~71% (35 GB cache, 209 GB model).  
Warm cache parallel pread: 1.0 ms; cold SSD: 5.8 ms; mixed: 2.4 ms.

---

## Experiment Results

### ✅ Kept: FMA Dequant Kernel (+12% → 4.36 tok/s)

Rearranged the inner loop of `dequant_matvec_4bit_v3` from:

```metal
acc += (float(nibble) * scale + bias) * x;
```

to:

```metal
float sx = scale * x, bx = bias * x;
acc += fma(float(nibble), sx, bx);
```

**Why it works:** Pre‑computing `scale*x` and `bias*x` per input element allows the GPU to use the fused multiply‑add unit for dequant+multiply in one instruction. Reduces per‑nibble cost from (convert + mul + add + mul + add) to (convert + fma + add).

**Impact:** `cmd1_wait` -5.4%, `cmd2_wait` -10.7%.  
**Total:** 3.90 → **4.36 tok/s**

---

### ❌ Discarded: LZ4 Expert Compression (-13%)

Repacked 209 GB of expert files to 175 GB with LZ4. Apple’s LZ4 decompressor runs at 41 GB/s (NEON hardware‑accelerated), making decompression only 0.17 ms per expert.

| Metric | Result |
|--------|--------|
| Isolated cold reads | 15‑24% faster |
| Isolated decompression | 0.17 ms (essentially free) |
| **Full pipeline** | **3.55 tok/s (-13%)** |

**Why it failed:** The 0.68 ms/layer decompress cost exceeds the warm cache I/O savings. The OS page cache is efficient enough that most reads are warm.

Also tested:
- LZFSE (2.6 GB/s, too slow)
- APFS transparent compression (kernel serialises read+decompress, 2× slower)
- Per‑expert files (15% slower from VFS metadata overhead)

**Key finding:** Apple’s M3 Max SSD is so fast that CPU‑based decompression cannot keep up for warm cache reads. LZ4 only wins for cold reads, but the page cache handles most reads.

---

### ❌ Discarded: Expert Routing Prediction (-18%)

Built a temporal prediction system: store previous token’s expert routing per layer, prefetch those experts into double‑buffered Metal buffers during the next token’s `CMD1_wait`.

| Metric | Value |
|--------|-------|
| Temporal hit rate | **25.6%** (only 1 of 4 experts matches) |
| P(all 4 hit) at 25% | 0.25⁴ = **0.4%** – practically zero |

**Why it failed:** 75% misses waste SSD bandwidth and require sync pread after the prediction wait. With K=4 parallel reads, wall time = max(4 reads). Need **all 4** to hit for improvement.

Also trained an MLP predictor (31% accuracy from pre‑attention hidden state – worse than temporal baseline). The `gate_proj` “logit lens” approach achieves 53% from pre‑attention state, but the K=4 exponential penalty still kills it.

---

### ❌ Discarded: `F_RDADVISE` Prefetch (net 0%)

Sent `F_RDADVISE` kernel hints between `CMD1` commit and wait to prefetch next token’s predicted experts during GPU compute.

| Metric | Change |
|--------|--------|
| `expert_io` | **-31%** (page cache warming works!) |
| `cmd2_wait` | **+73%** (GPU memory bandwidth contention) |
| **Net** | **0% across 5 diverse prompts** |

**Root cause:** Apple Silicon unified memory architecture. SSD DMA and GPU matvec share the same memory controller. The GPU’s dequant kernels are bandwidth‑saturated at 418 GiB/s. Even 17.5 GB/s of background DMA (~4%) causes disproportionate latency spikes. This is **architectural** – cannot be worked around in software.

---

### ❌ Discarded: GPU Kernel Variants

| Variant | Result | Reason |
|---------|--------|--------|
| LUT dequant (v5) | -2% | Indirect register access serialises |
| Vector load (v4) | -3% | Register pressure |
| `extract_bits` intrinsic | Neutral | Compiler already generates same instruction |
| Spin‑poll GPU wait | -23% | CPU spinning steals thermal budget |
| `addCompletedHandler` | Neutral | Real workloads hide wait overhead |

---

### ❌ Discarded: I/O Path Alternatives

| Alternative | Result | Reason |
|-------------|--------|--------|
| `dispatch_io` | -70% | `dispatch_data` management overhead |
| `aio_read` | -7% | Matches GCD group + pread |
| Expert file clustering | 0% | NVMe ignores scatter at 7 MB granularity |
| GPU private buffer compression | -20% | Blit cost > matvec savings |

---

### Analyzed but not implemented: MTP Speculative Decoding

Qwen 3.5 ships with an MTP (Multi‑Token Prediction) head – a single MoE transformer layer that predicts the next‑next token. The head exists in the model config (`mtp_num_hidden_layers: 1`) but weights were stripped from the MLX quantisation.

**Analysis:** MTP speculative decoding does **not** help for MoE with SSD streaming. Each speculated token requires its **own** expert routing and I/O. Batched verification of 2 tokens costs ~1.75× expert I/O for 1.7 tokens (70% acceptance). Break‑even at best.

This contrasts with dense models where verification cost is constant regardless of batch size (same weights for every token).

---

## The Unified Memory Constraint

**Single most important finding:** On Apple Silicon, SSD DMA and GPU compute **cannot be profitably overlapped**. They share the same memory controller, and the GPU’s dequant kernels are bandwidth‑saturated. Any background I/O during GPU compute causes disproportionate GPU slowdown.

**Implication:** The serial pipeline (GPU → SSD → GPU) is actually **hardware‑optimal** for this architecture. The current pipeline already achieves the best possible scheduling.

---

## Summary

| Configuration | tok/s | Status |
|---------------|-------|--------|
| 2‑bit experts (best speed) | 5.74 | ❌ Quality regression (broken JSON) |
| 2‑bit peak single token | 7.05 | Warm cache burst |
| **4‑bit + FMA kernel** | **4.36** | ✅ **Current best – quality preserved** |
| 4‑bit baseline (no FMA) | 3.90 | Previous 4‑bit baseline |
| 4‑bit + LZ4 compression | 3.55 | Decompress overhead > I/O savings |
| 4‑bit + temporal prediction | 3.18 | 25% hit rate wastes SSD bandwidth |
| 4‑bit + `F_RDADVISE` prefetch | 3.91 | GPU contention cancels I/O savings |

---

## Performance Ceiling

The 4‑bit performance ceiling on M3 Max 48 GB is approximately **4.4 tok/s** for sustained generation, limited by:

| Bottleneck | Time | % |
|------------|------|---|
| SSD expert I/O | 2.4 ms/layer | 56% |
| GPU dequant matvec | 1.8 ms/layer | 41% |
| CPU overhead | 0.1 ms/layer | 3% |

**Further improvement requires either:**
- Hardware changes (more RAM for expert caching, faster SSD)
- Model architecture changes (fewer/smaller experts, larger shared expert to reduce per‑token I/O)

