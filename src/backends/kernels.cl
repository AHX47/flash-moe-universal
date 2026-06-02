/*
 * kernels.cl — OpenCL compute kernels for Flash-MoE Universal
 *
 * Cross-platform GPU kernels:
 *   - AMD GPUs (ROCm/OpenCL)
 *   - Intel iGPU/dGPU (OpenCL)
 *   - NVIDIA (OpenCL)
 *   - Apple GPU on Intel Macs (OpenCL)
 *
 * These are the same algorithms as shaders.metal but in OpenCL C.
 */

// ─── 4-bit dequantize + matrix-vector multiply ──────────────────────────────
__kernel void dequant_matvec_4bit(
    __global const uchar  *W_packed,   // [rows * cols/2]
    __global const ushort *scales,     // [rows * groups] fp16
    __global const ushort *biases,     // [rows * groups] fp16
    __global const float  *x,          // [cols]
    __global       float  *out,        // [rows]
    int rows, int cols, int group_size)
{
    int r = get_global_id(0);
    if (r >= rows) return;

    int groups = cols / group_size;
    float acc = 0.0f;

    __global const uchar  *w_row = W_packed + (long)r * (cols / 2);
    __global const ushort *s_row = scales   + (long)r * groups;
    __global const ushort *b_row = biases   + (long)r * groups;

    for (int g = 0; g < groups; g++) {
        // Decode fp16 scale/bias
        float sc = vload_half(g, (__global const half*)s_row);
        float bi = vload_half(g, (__global const half*)b_row);
        int base = g * group_size;

        for (int k = 0; k < group_size; k += 2) {
            int j = base + k;
            uchar byte = w_row[j / 2];
            float v0 = (float)(byte & 0xF) * sc + bi;
            float v1 = (float)(byte >> 4)  * sc + bi;
            acc += v0 * x[j];
            acc += v1 * x[j + 1];
        }
    }
    out[r] = acc;
}

// ─── 4-bit dequant matvec with local memory reduction (optimized) ────────────
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void dequant_matvec_4bit_fast(
    __global const uchar  *W_packed,
    __global const ushort *scales,
    __global const ushort *biases,
    __global const float  *x,
    __global       float  *out,
    int rows, int cols, int group_size)
{
    int r   = get_group_id(0);
    int tid = get_local_id(0);
    if (r >= rows) return;

    __local float lmem[64];

    int groups = cols / group_size;
    float acc  = 0.0f;

    __global const uchar  *w_row = W_packed + (long)r * (cols / 2);
    __global const ushort *s_row = scales   + (long)r * groups;
    __global const ushort *b_row = biases   + (long)r * groups;

    // Each work-item handles a subset of groups
    for (int g = tid; g < groups; g += 64) {
        float sc = vload_half(g, (__global const half*)s_row);
        float bi = vload_half(g, (__global const half*)b_row);
        int base = g * group_size;
        for (int k = 0; k < group_size; k += 2) {
            int j = base + k;
            uchar byte = w_row[j / 2];
            acc += ((float)(byte & 0xF) * sc + bi) * x[j];
            acc += ((float)(byte >> 4)  * sc + bi) * x[j + 1];
        }
    }

    lmem[tid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction
    for (int s = 32; s > 0; s >>= 1) {
        if (tid < s) lmem[tid] += lmem[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) out[r] = lmem[0];
}

// ─── 2-bit dequantize + matvec ───────────────────────────────────────────────
__kernel void dequant_matvec_2bit(
    __global const uchar *W_packed,    // [rows * cols/4]
    __global const float *scales,      // [rows * groups] float32
    __global const float *biases,
    __global const float *x,
    __global       float *out,
    int rows, int cols, int group_size)
{
    int r = get_global_id(0);
    if (r >= rows) return;

    int groups = cols / group_size;
    float acc  = 0.0f;

    __global const uchar *w_row = W_packed + (long)r * (cols / 4);
    __global const float *s_row = scales   + (long)r * groups;
    __global const float *b_row = biases   + (long)r * groups;

    for (int g = 0; g < groups; g++) {
        float sc = s_row[g];
        float bi = b_row[g];
        int base = g * group_size;
        for (int k = 0; k < group_size; k += 4) {
            int j = base + k;
            uchar byte = w_row[j / 4];
            acc += ((float)((byte >> 0) & 3) * sc + bi) * x[j];
            acc += ((float)((byte >> 2) & 3) * sc + bi) * x[j + 1];
            acc += ((float)((byte >> 4) & 3) * sc + bi) * x[j + 2];
            acc += ((float)((byte >> 6) & 3) * sc + bi) * x[j + 3];
        }
    }
    out[r] = acc;
}

// ─── RMS Normalization ───────────────────────────────────────────────────────
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void rms_norm(
    __global const float *x,
    __global const float *weight,
    __global       float *out,
    int n, float eps)
{
    __local float lmem[256];
    int tid = get_local_id(0);

    float ss = 0.0f;
    for (int i = tid; i < n; i += 256) ss += x[i] * x[i];
    lmem[tid] = ss;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) lmem[tid] += lmem[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float scale = rsqrt(lmem[0] / n + eps);
    for (int i = tid; i < n; i += 256)
        out[i] = x[i] * scale * weight[i];
}

// ─── SwiGLU fused activation ─────────────────────────────────────────────────
__kernel void swiglu_fused(
    __global const float *gate,
    __global const float *up,
    __global       float *out,
    int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float g = gate[i];
    float silu = g / (1.0f + exp(-g));
    out[i] = silu * up[i];
}

// ─── Softmax (single row) ────────────────────────────────────────────────────
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void softmax(
    __global float *x,
    int n)
{
    __local float lmem[256];
    int tid = get_local_id(0);

    // Find max
    float maxv = -INFINITY;
    for (int i = tid; i < n; i += 256) maxv = max(maxv, x[i]);
    lmem[tid] = maxv;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) lmem[tid] = max(lmem[tid], lmem[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxv = lmem[0];

    // Exp and sum
    float sumv = 0.0f;
    for (int i = tid; i < n; i += 256) {
        x[i] = exp(x[i] - maxv);
        sumv += x[i];
    }
    lmem[tid] = sumv;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) lmem[tid] += lmem[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv_sum = 1.0f / lmem[0];
    for (int i = tid; i < n; i += 256) x[i] *= inv_sum;
}

// ─── Weighted sum of expert outputs (MoE combine) ────────────────────────────
__kernel void moe_combine(
    __global const float *expert_outs,  // [K * hidden_size]
    __global const float *weights,      // [K] router weights
    __global const float *residual,     // [hidden_size]
    __global       float *out,          // [hidden_size]
    int K, int hidden_size)
{
    int i = get_global_id(0);
    if (i >= hidden_size) return;

    float sum = residual[i];
    for (int k = 0; k < K; k++)
        sum += weights[k] * expert_outs[k * hidden_size + i];
    out[i] = sum;
}

// ─── Dot-product attention scores ────────────────────────────────────────────
__kernel void attn_scores(
    __global const float *q,       // [num_heads * head_dim]
    __global const float *k_cache, // [seq_len * num_kv_heads * head_dim]
    __global       float *scores,  // [num_heads * seq_len]
    int num_heads, int num_kv_heads, int head_dim, int seq_len)
{
    int h   = get_global_id(0);  // head index
    int t   = get_global_id(1);  // token index
    if (h >= num_heads || t >= seq_len) return;

    int kv_h = h * num_kv_heads / num_heads;
    __global const float *q_h = q       + h   * head_dim;
    __global const float *k_t = k_cache + t   * num_kv_heads * head_dim + kv_h * head_dim;

    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) dot += q_h[d] * k_t[d];
    scores[h * seq_len + t] = dot * rsqrt((float)head_dim);
}
