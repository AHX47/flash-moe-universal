/*
 * cpu_backend.h — CPU inference backend for Flash-MoE Universal
 *
 * Supports all architectures:
 *   - x86_64 : AVX2, AVX512, SSE4.1, scalar fallback
 *   - ARM64  : NEON
 *   - Generic: Pure C scalar
 *
 * Key operations:
 *   - 4-bit dequantize + matvec  (core bottleneck)
 *   - 2-bit dequantize + matvec
 *   - RMS normalization
 *   - SwiGLU activation
 *   - Softmax
 *   - RoPE position encoding
 */

#pragma once

#include "../platform/platform.h"
#include "../core/model_config.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BLAS detection (for large matrix ops like lm_head projection)
// ============================================================================

#if defined(OS_MACOS)
  #include <Accelerate/Accelerate.h>
  #define HAS_BLAS 1
#elif defined(__has_include)
  #if __has_include(<cblas.h>)
    #include <cblas.h>
    #define HAS_BLAS 1
  #elif __has_include(<openblas/cblas.h>)
    #include <openblas/cblas.h>
    #define HAS_BLAS 1
  #endif
#endif

// ============================================================================
// Half-precision float support
// ============================================================================

typedef uint16_t float16_t;

static inline float fp16_to_fp32(float16_t h) {
    uint32_t t1 = ((uint32_t)(h & 0x7FFF)) << 13;
    uint32_t t2 = ((uint32_t)(h & 0x8000)) << 16;
    uint32_t t3 = t1 + 0x38000000U;
    uint32_t t4 = t3 + (((t1 >= 0x0C000000U) ? 0x38000000U : 0U));
    uint32_t result = t4 | t2;
    float f;
    memcpy(&f, &result, 4);
    return f;
}

// ============================================================================
// 4-bit packed weight layout (matches original flash-moe)
//
// Each expert: [gate_w | gate_scales | gate_biases | up_w | ... | down_w | ...]
//   gate_w:      out_dim * (in_dim / 2) bytes  (2 nibbles per byte)
//   gate_scales: out_dim * (in_dim / group_size) * sizeof(float16) bytes
//   gate_biases: same as scales
//   (same for up and down)
//
// Group-quantized: for every GROUP_SIZE inputs, one (scale, bias) pair.
// dequant: val = (nibble * scale + bias)
// matvec:  out[i] += dequant(W[i,j]) * x[j]  for all j
// ============================================================================

#define CPU_BACKEND_GROUP_SIZE 64

// ============================================================================
// Scalar fallback — works on every CPU
// ============================================================================

static inline void cpu_dequant_matvec_4bit_scalar(
    const uint8_t  *W_packed,  // rows * cols/2 bytes (row-major nibbles)
    const uint16_t *scales,    // rows * (cols/group_size) fp16
    const uint16_t *biases,    // rows * (cols/group_size) fp16
    const float    *x,         // [cols]
    float          *out,       // [rows]
    int             rows,
    int             cols,
    int             group_size)
{
    int groups_per_row = cols / group_size;
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        const uint8_t  *w_row = W_packed + r * (cols / 2);
        const uint16_t *s_row = scales  + r * groups_per_row;
        const uint16_t *b_row = biases  + r * groups_per_row;
        for (int g = 0; g < groups_per_row; g++) {
            float sc = fp16_to_fp32(s_row[g]);
            float bi = fp16_to_fp32(b_row[g]);
            int base = g * group_size;
            for (int k = 0; k < group_size; k += 2) {
                int j = base + k;
                uint8_t byte = w_row[j / 2];
                float v0 = (float)(byte & 0xF)  * sc + bi;
                float v1 = (float)(byte >> 4)    * sc + bi;
                acc += v0 * x[j];
                acc += v1 * x[j + 1];
            }
        }
        out[r] += acc;
    }
}

static inline void cpu_dequant_matvec_2bit_scalar(
    const uint8_t  *W_packed,  // rows * cols/4 bytes (4 x 2-bit per byte)
    const float    *scales,    // rows * (cols/group_size)
    const float    *biases,    // rows * (cols/group_size)
    const float    *x,
    float          *out,
    int             rows,
    int             cols,
    int             group_size)
{
    int groups_per_row = cols / group_size;
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        const uint8_t *w_row = W_packed + r * (cols / 4);
        const float   *s_row = scales   + r * groups_per_row;
        const float   *b_row = biases   + r * groups_per_row;
        for (int g = 0; g < groups_per_row; g++) {
            float sc = s_row[g];
            float bi = b_row[g];
            int base = g * group_size;
            for (int k = 0; k < group_size; k += 4) {
                int j = base + k;
                uint8_t byte = w_row[j / 4];
                acc += ((float)((byte >> 0) & 3) * sc + bi) * x[j];
                acc += ((float)((byte >> 2) & 3) * sc + bi) * x[j+1];
                acc += ((float)((byte >> 4) & 3) * sc + bi) * x[j+2];
                acc += ((float)((byte >> 6) & 3) * sc + bi) * x[j+3];
            }
        }
        out[r] += acc;
    }
}

// ============================================================================
// AVX2 accelerated 4-bit dequant matvec (x86_64)
// ============================================================================

#ifdef HAS_AVX2

static inline void cpu_dequant_matvec_4bit_avx2(
    const uint8_t  *W_packed,
    const uint16_t *scales,
    const uint16_t *biases,
    const float    *x,
    float          *out,
    int             rows,
    int             cols,
    int             group_size)
{
    int groups_per_row = cols / group_size;
    const __m256i lo_mask = _mm256_set1_epi8(0x0F);

    for (int r = 0; r < rows; r++) {
        const uint8_t  *w_row = W_packed + r * (cols / 2);
        const uint16_t *s_row = scales   + r * groups_per_row;
        const uint16_t *b_row = biases   + r * groups_per_row;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        for (int g = 0; g < groups_per_row; g++) {
            float sc = fp16_to_fp32(s_row[g]);
            float bi = fp16_to_fp32(b_row[g]);
            __m256 sc_v = _mm256_set1_ps(sc);
            __m256 bi_v = _mm256_set1_ps(bi);
            int base = g * group_size;

            // Process 32 elements (16 bytes packed) per iteration
            for (int k = 0; k < group_size; k += 32) {
                int j = base + k;
                // Load 16 bytes = 32 nibbles
                __m128i packed = _mm_loadu_si128((const __m128i*)(w_row + j/2));
                __m256i packed_256 = _mm256_cvtepu8_epi16(packed);

                // Split nibbles
                __m256i lo_nibbles = _mm256_and_si256(packed_256, _mm256_set1_epi16(0x000F));
                __m256i hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(packed_256, 4),
                                                       _mm256_set1_epi16(0x000F));

                // Convert to float (lo nibbles -> first 8 floats of pair)
                __m128i lo_lo = _mm256_castsi256_si128(lo_nibbles);
                __m128i lo_hi = _mm256_extracti128_si256(lo_nibbles, 1);
                __m256  lo_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_lo));
                __m256  lo_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo_hi));

                // Dequant: val = nibble * scale + bias (FMA: fma(nibble, scale, bias))
                lo_f0 = _mm256_fmadd_ps(lo_f0, sc_v, bi_v);
                lo_f1 = _mm256_fmadd_ps(lo_f1, sc_v, bi_v);

                // Load x
                __m256 x0 = _mm256_loadu_ps(x + j);
                __m256 x1 = _mm256_loadu_ps(x + j + 8);

                // Accumulate with FMA
                acc0 = _mm256_fmadd_ps(lo_f0, x0, acc0);
                acc1 = _mm256_fmadd_ps(lo_f1, x1, acc1);

                // Hi nibbles for j+16..j+31
                __m128i hi_lo = _mm256_castsi256_si128(hi_nibbles);
                __m128i hi_hi = _mm256_extracti128_si256(hi_nibbles, 1);
                __m256  hi_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_lo));
                __m256  hi_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi_hi));

                hi_f0 = _mm256_fmadd_ps(hi_f0, sc_v, bi_v);
                hi_f1 = _mm256_fmadd_ps(hi_f1, sc_v, bi_v);

                __m256 x2 = _mm256_loadu_ps(x + j + 16);
                __m256 x3 = _mm256_loadu_ps(x + j + 24);

                acc0 = _mm256_fmadd_ps(hi_f0, x2, acc0);
                acc1 = _mm256_fmadd_ps(hi_f1, x3, acc1);
            }
        }
        // Horizontal reduce
        __m256 sum256 = _mm256_add_ps(acc0, acc1);
        __m128 lo128  = _mm256_castps256_ps128(sum256);
        __m128 hi128  = _mm256_extractf128_ps(sum256, 1);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        out[r] += _mm_cvtss_f32(sum128);
    }
}
#endif /* HAS_AVX2 */

// ============================================================================
// ARM NEON accelerated 4-bit dequant matvec
// ============================================================================

#ifdef HAS_NEON
static inline void cpu_dequant_matvec_4bit_neon(
    const uint8_t  *W_packed,
    const uint16_t *scales,
    const uint16_t *biases,
    const float    *x,
    float          *out,
    int             rows,
    int             cols,
    int             group_size)
{
    int groups_per_row = cols / group_size;
    for (int r = 0; r < rows; r++) {
        const uint8_t  *w_row = W_packed + r * (cols / 2);
        const uint16_t *s_row = scales   + r * groups_per_row;
        const uint16_t *b_row = biases   + r * groups_per_row;
        float32x4_t acc = vdupq_n_f32(0.0f);

        for (int g = 0; g < groups_per_row; g++) {
            float sc = fp16_to_fp32(s_row[g]);
            float bi = fp16_to_fp32(b_row[g]);
            float32x4_t sc_v = vdupq_n_f32(sc);
            float32x4_t bi_v = vdupq_n_f32(bi);
            int base = g * group_size;

            for (int k = 0; k < group_size; k += 8) {
                int j = base + k;
                // Load 4 bytes = 8 nibbles
                uint8x8_t packed = vld1_u8(w_row + j/2);
                uint8x8_t lo = vand_u8(packed, vdup_n_u8(0x0F));
                uint8x8_t hi = vshr_n_u8(packed, 4);

                // lo nibbles -> floats
                uint16x8_t lo16 = vmovl_u8(lo);
                uint32x4_t lo32_0 = vmovl_u16(vget_low_u16(lo16));
                float32x4_t lo_f = vcvtq_f32_u32(lo32_0);
                lo_f = vmlaq_f32(bi_v, lo_f, sc_v);  // FMA: bias + nibble*scale
                float32x4_t x_v = vld1q_f32(x + j);
                acc = vmlaq_f32(acc, lo_f, x_v);

                uint32x4_t lo32_1 = vmovl_u16(vget_high_u16(lo16));
                float32x4_t lo_f2 = vcvtq_f32_u32(lo32_1);
                lo_f2 = vmlaq_f32(bi_v, lo_f2, sc_v);
                float32x4_t x_v2 = vld1q_f32(x + j + 4);
                acc = vmlaq_f32(acc, lo_f2, x_v2);
                (void)hi;
            }
        }
        // Horizontal add
        float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        out[r] += vget_lane_f32(vpadd_f32(sum2, sum2), 0);
    }
}
#endif /* HAS_NEON */

// ============================================================================
// Dispatcher — picks best available implementation at compile time
// ============================================================================

static inline void cpu_dequant_matvec_4bit(
    const uint8_t  *W_packed,
    const uint16_t *scales,
    const uint16_t *biases,
    const float    *x,
    float          *out,
    int             rows,
    int             cols,
    int             group_size)
{
    // Zero output first
    memset(out, 0, rows * sizeof(float));

#if defined(HAS_AVX2)
    cpu_dequant_matvec_4bit_avx2(W_packed, scales, biases, x, out, rows, cols, group_size);
#elif defined(HAS_NEON)
    cpu_dequant_matvec_4bit_neon(W_packed, scales, biases, x, out, rows, cols, group_size);
#else
    cpu_dequant_matvec_4bit_scalar(W_packed, scales, biases, x, out, rows, cols, group_size);
#endif
}

// ============================================================================
// RMS Normalization
// ============================================================================

static inline void cpu_rms_norm(
    const float *x,
    const float *weight,
    float       *out,
    int          n,
    float        eps)
{
    float ss = 0.0f;

#ifdef HAS_AVX2
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    ss = _mm_cvtss_f32(sum);
    for (; i < n; i++) ss += x[i] * x[i];
#elif defined(HAS_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vmlaq_f32(acc, v, v);
    }
    float32x2_t s = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    ss = vget_lane_f32(vpadd_f32(s, s), 0);
    for (; i < n; i++) ss += x[i] * x[i];
#else
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
#endif

    float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

// ============================================================================
// SwiGLU Activation: out[i] = gate[i] * silu(up[i])
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================================

static inline void cpu_swiglu(
    const float *gate,
    const float *up,
    float       *out,
    int          n)
{
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * u;
    }
}

// ============================================================================
// Softmax (in-place)
// ============================================================================

static inline void cpu_softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

// ============================================================================
// TopK (in-place argsort, returns indices of K largest values)
// ============================================================================

static inline void cpu_topk(const float *scores, int n, int k, int *indices) {
    // Simple O(n*k) selection — fast enough for k=4, n≤512
    for (int i = 0; i < k; i++) indices[i] = -1;
    for (int i = 0; i < k; i++) {
        float best = -1e38f;
        for (int j = 0; j < n; j++) {
            bool already = false;
            for (int ii = 0; ii < i; ii++) if (indices[ii] == j) { already = true; break; }
            if (!already && scores[j] > best) { best = scores[j]; indices[i] = j; }
        }
    }
}

// ============================================================================
// Embedding lookup
// ============================================================================

static inline void cpu_embedding_lookup(
    const uint8_t  *embed_weights, // [vocab_size * hidden_size / 2] 4-bit packed
    const uint16_t *embed_scales,
    const uint16_t *embed_biases,
    int             token_id,
    float          *out,
    int             hidden_size,
    int             group_size)
{
    // Each row = one embedding vector, 4-bit packed
    const uint8_t  *w = embed_weights + (size_t)token_id * (hidden_size / 2);
    const uint16_t *s = embed_scales  + (size_t)token_id * (hidden_size / group_size);
    const uint16_t *b = embed_biases  + (size_t)token_id * (hidden_size / group_size);
    int groups = hidden_size / group_size;
    for (int g = 0; g < groups; g++) {
        float sc = fp16_to_fp32(s[g]);
        float bi = fp16_to_fp32(b[g]);
        int base = g * group_size;
        for (int k = 0; k < group_size; k += 2) {
            int j = base + k;
            uint8_t byte = w[j / 2];
            out[j]   = (float)(byte & 0xF) * sc + bi;
            out[j+1] = (float)(byte >> 4)  * sc + bi;
        }
    }
}

// ============================================================================
// RoPE (Rotary Position Embedding)
// ============================================================================

static inline void cpu_rope_embed(
    float       *q,           // [num_heads * head_dim]
    float       *k,           // [num_kv_heads * head_dim]
    int          pos,
    int          num_q_heads,
    int          num_k_heads,
    int          head_dim,
    int          rotary_dim,  // partial rotary (rotary_dim <= head_dim)
    float        theta)
{
    for (int h = 0; h < num_q_heads + num_k_heads; h++) {
        float *vec = (h < num_q_heads) ? (q + h * head_dim) : (k + (h - num_q_heads) * head_dim);
        for (int i = 0; i < rotary_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2*i) / rotary_dim);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float x0 = vec[i];
            float x1 = vec[i + rotary_dim/2];
            vec[i]              = x0 * cos_a - x1 * sin_a;
            vec[i + rotary_dim/2] = x0 * sin_a + x1 * cos_a;
        }
    }
}

// ============================================================================
// Parallel matrix-vector multiply (multithreaded, for large projections)
// ============================================================================

typedef struct {
    const uint8_t  *W_packed;
    const uint16_t *scales;
    const uint16_t *biases;
    const float    *x;
    float          *out;
    int             rows;
    int             cols;
    int             group_size;
    int             row_start;
    int             row_end;
} ParallelMatvecArgs;

#ifndef OS_WINDOWS
static void* parallel_matvec_worker(void *arg) {
    ParallelMatvecArgs *a = (ParallelMatvecArgs*)arg;
    int len = a->row_end - a->row_start;
    if (len <= 0) return NULL;
    const uint8_t  *w = a->W_packed + (size_t)a->row_start * (a->cols / 2);
    const uint16_t *s = a->scales   + (size_t)a->row_start * (a->cols / a->group_size);
    const uint16_t *b = a->biases   + (size_t)a->row_start * (a->cols / a->group_size);
    float          *o = a->out      + a->row_start;
    cpu_dequant_matvec_4bit(w, s, b, a->x, o, len, a->cols, a->group_size);
    return NULL;
}
#endif

static inline void cpu_parallel_dequant_matvec_4bit(
    const uint8_t  *W_packed,
    const uint16_t *scales,
    const uint16_t *biases,
    const float    *x,
    float          *out,
    int             rows,
    int             cols,
    int             group_size,
    int             num_threads)
{
    if (num_threads <= 1) {
        cpu_dequant_matvec_4bit(W_packed, scales, biases, x, out, rows, cols, group_size);
        return;
    }

#ifndef OS_WINDOWS
    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ParallelMatvecArgs *args = (ParallelMatvecArgs*)malloc(num_threads * sizeof(ParallelMatvecArgs));

    int rows_per_thread = rows / num_threads;
    for (int t = 0; t < num_threads; t++) {
        args[t].W_packed  = W_packed;
        args[t].scales    = scales;
        args[t].biases    = biases;
        args[t].x         = x;
        args[t].out       = out;
        args[t].rows      = rows;
        args[t].cols      = cols;
        args[t].group_size = group_size;
        args[t].row_start = t * rows_per_thread;
        args[t].row_end   = (t == num_threads-1) ? rows : (t+1) * rows_per_thread;
        pthread_create(&threads[t], NULL, parallel_matvec_worker, &args[t]);
    }
    for (int t = 0; t < num_threads; t++) pthread_join(threads[t], NULL);
    free(threads);
    free(args);
#else
    cpu_dequant_matvec_4bit(W_packed, scales, biases, x, out, rows, cols, group_size);
#endif
}

// ============================================================================
// CPU capability detection
// ============================================================================

typedef struct {
    bool has_avx512;
    bool has_avx2;
    bool has_avx;
    bool has_sse41;
    bool has_neon;
    int  num_cores;
    char arch_name[32];
    char simd_name[32];
} CPUCapabilities;

static inline CPUCapabilities cpu_detect(void) {
    CPUCapabilities caps = {0};
    caps.num_cores = platform_cpu_count();
    strncpy(caps.arch_name, ARCH_NAME, 31);

#ifdef ARCH_X86_64
    #if defined(HAS_AVX512)
        caps.has_avx512 = true; strncpy(caps.simd_name, "AVX-512", 31);
    #elif defined(HAS_AVX2)
        caps.has_avx2   = true; strncpy(caps.simd_name, "AVX2+FMA", 31);
    #elif defined(HAS_AVX)
        caps.has_avx    = true; strncpy(caps.simd_name, "AVX", 31);
    #elif defined(HAS_SSE41)
        caps.has_sse41  = true; strncpy(caps.simd_name, "SSE4.1", 31);
    #else
        strncpy(caps.simd_name, "Scalar", 31);
    #endif
#endif
#ifdef HAS_NEON
    caps.has_neon = true;
    strncpy(caps.simd_name, "NEON", 31);
#endif
#ifdef ARCH_GENERIC
    strncpy(caps.simd_name, "Scalar", 31);
#endif
    return caps;
}

#ifdef __cplusplus
}
#endif
