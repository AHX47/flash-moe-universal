
# Cross‑Layer Async Pread Pipeline – Design & Analysis

## Status

- Async pread mechanism implemented and working (`async_pread_start`/`wait` in `infer.m`).
- Within‑layer overlap tested: **no improvement** (only 0.1 ms overlap window).
- **Goal:** Cross‑layer overlap to hide ~2 ms of pread time.

---

## Current Per‑Layer Sequence (4.5 ms total)

```
[deferred_wait] → [CMD1 submit+wait] → [CPU attn] → [CMD2 submit+wait] → [routing] → [SYNC pread] → [CMD3 submit]
     0.87ms           0.5ms              0.27ms          0.45ms           0.003ms      2.43ms          0.03ms
```

**Expert I/O dominates:** 2.43 ms (54%) of the layer time.

---

## Target Sequence

We want the pread for layer **N** to run while layer **N+1** is doing its compute.

```
Layer N:      ... → [routing] → [START async pread into BUF_A] → [CMD3 submit (using BUF_B from prev)]
Layer N+1:    [deferred_wait] → [CMD1] → [CPU attn] → [CMD2] → [routing] → [WAIT async pread BUF_A] → [CMD3 submit (using BUF_A)]
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                    ~2.1ms of compute overlap with N's pread
```

By the time layer N+1 needs its expert data, the pread for layer N has had **2.1 ms head start**.

---

## The Fundamental Problem

We cannot start the pread for layer N until **routing** is done, because we need to know *which* experts to load. Routing is the **last** step before the pread in the current pipeline.

So we cannot overlap pread with **the same layer's** CMD1+attn+CMD2 – we don’t have the expert indices yet.

---

## Potential Solutions (Evaluated)

### 1. Decouple Routing from Expert Loading

Split CMD2 into:
- `CMD2a`: o_proj + residual + norm → produces `h_post`
- `CMD2b`: routing gate_proj → produces gate scores

Then:
```
Layer N:   CMD1 → CPU attn → CMD2a+CMD2b → wait → routing topK → [START async pread] → CMD3
```
❌ Still doesn’t help – pread still starts after routing, no overlap within the same layer.

---

### 2. Pipeline Expert Data One Layer Ahead

At the end of layer N, we already have N’s expert data loaded. Submit CMD3_N (deferred) and **simultaneously start pread for layer N+1**.

```
Layer N end:   [submit CMD3_N using BUF_A] + [start async pread for N+1 into BUF_B]
Layer N+1:     [deferred_wait N] → [CMD1] → [attn] → [CMD2] → [routing]
               [async pread N+1 completes during this time]
               → [match? use BUF_B; else sync pread]
```

❌ We don’t know layer N+1’s experts at the end of layer N. Requires **prediction**.

Previous prediction attempts failed (53% accuracy, overhead > benefit).

---

### 3. Overlap Pread with CMD3 GPU Execution

CMD3 runs on GPU while CPU starts the next layer. Why not start the **next layer’s expert pread** during that time?

```
[submit CMD3_N] → [start next layer's CMD1+attn+CMD2] → [routing N+1] → [pread N+1]
                  ↑ CMD3_N runs on GPU here, overlapping with N+1's compute
```

❌ The pread for N+1 still starts **after** routing N+1, which is after CMD2_N+1, which waits for CMD3_N. No overlap gained.

---

### 4. ✅ Start Pread During CMD1 Wait (Temporal Prediction)

**Insight:** During `CMD1_wait` (0.87 ms), the CPU is **idle** – we could use that time to pre‑load experts **predicted from the previous token** for the same layer.

```
Layer N, token T:
  [CMD1 submit] → [while waiting: pread PREDICTED experts into BUF_B based on token T-1]
  → [CMD1 wait returns] → [CPU attn] → [CMD2] → [routing]
  → [check predictions: how many of K=4 match?]
  → [pread only the MISSES into BUF_A (typically 2‑3 instead of 4)]
  → [CMD3 using mix of BUF_A (misses) and BUF_B (hits)]
```

**Expected gains:**
- 30% hit rate → 1.2 of 4 experts pre‑loaded → saves ~30% of pread time
- 50% hit rate → 2 of 4 pre‑loaded → saves ~50%

**Why this is different from previous failed attempts:**
- ✅ Loading into **scratch buffers** (no cache pollution)
- ✅ Using **idle CPU time** (`CMD1_wait` – no additional cost)
- ✅ Only predicting from **previous token at same layer** (simple, no extra `gate_proj` if we reuse stored indices)
- ✅ Only need to sync‑pread the **misses**, not all K experts

**Cost:** `gate_proj` matvec (~0.1 ms) + predicted pread during idle time → net ~0.1 ms per layer.

---

## Implementation Plan

### Double‑Buffer Expert Data

We already have two sets of Metal buffers:
- `buf_multi_expert_data[MAX_K]` (set A)
- `buf_multi_expert_data_B[MAX_K]` (set B)

Add a flip flag to alternate which set is written to and which is read from.

### Modify `fused_layer_forward()`

1. At layer start, if an async pread from previous token is in flight, **do not wait** yet.
2. Submit CMD1.
3. During `CMD1_wait`, start async pread of **predicted experts** (from previous token) into the **other** buffer set.
4. Continue with CPU attention, CMD2, routing.
5. After routing, compare actual experts with predicted ones.
6. For hits, mark those experts as already loaded.
7. For misses, synchronously load them into the **current** buffer set.
8. Submit CMD3 using the combined buffer set.

---

## Key Files

| File | Relevant lines |
|------|----------------|
| `infer.m` | `fused_layer_forward()` (~4900‑5230) |
| `infer.m` | `async_pread_start`/`wait` (~3011‑3050) |
| `infer.m` | `g_prefetch_experts` (~195) – temporal prediction state |
| `MetalCtx` | `buf_multi_expert_data` (A) and `buf_multi_expert_data_B` (B) |

---

## Baseline & Target

| Metric | Baseline (4‑bit, K=4, Trust OS) | Target |
|--------|----------------------------------|--------|
| Tok/s | 3.50‑3.70 | 4.5‑5.0 |
| `expert_io` | 2.43 ms/layer | ~1.5 ms/layer |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-------------|
| SSD bandwidth contention between predicted and actual reads | Predicted reads run during idle CPU time, actual reads only for misses – total I/O less than baseline |
| Previous speculative attempts failed | This uses scratch buffers (no cache pollution) and only predicts from previous token (temporal locality ~25% hit rate may still be marginal) |
| Double‑buffer complexity | Well‑understood pattern; add explicit state machine |
| Quality regression | Validate by comparing output tokens with and without optimisation |

---

## Previous Attempts (All Failed)

| Approach | Result | Why |
|----------|--------|-----|
| Speculative early routing on pre‑attention state | 53% accuracy, but **38% slower** | Cache pollution from wrong predictions |
| `F_RDADVISE` hints | net 0% | GPU contention (SSD DMA slowed GPU) |
| Temporal `F_RDADVISE` with lead time | 65‑80% wrong predictions → slower | Most predictions wrong, wasted bandwidth |
| `mmap` + `memcpy` | **5.5× slower** for cold data | Page fault overhead |

**What’s different this time:**
- Loading into **scratch buffers** (no cache pollution)
- Using **CMD1_wait idle time** (no additional CPU cost)
- Only predicting from previous token at same layer (simple, no `gate_proj` overhead if we reuse stored indices)
- Only need to sync‑pread the **misses**, not all 4 experts

---

## Conclusion

The most promising path forward is **temporal prediction + double‑buffered scratch loads**, overlapping pread with the idle `CMD1_wait` period. This avoids the cache pollution and GPU contention that killed previous attempts.

If temporal hit rate proves too low (<25%), the only remaining option would be hardware changes (faster SSD, more RAM) or model‑level changes (fewer experts, larger shared expert).

