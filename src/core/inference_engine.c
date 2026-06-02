/*
 * inference_engine.c — Universal Flash-MoE inference engine
 *
 * Cross-platform implementation:
 *   - Reads model weights from binary files (same format as flash-moe)
 *   - Runs transformer forward pass on CPU (with SIMD acceleration)
 *   - Streams expert weights from disk per-layer (like flash-moe)
 *   - Supports all model sizes via dynamic ModelConfig
 *   - Optionally calls Metal / OpenCL backend if available
 */

#include "inference_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdarg.h>

#ifdef OS_WINDOWS
  #include <io.h>
#else
  #include <unistd.h>
  #include <pthread.h>
#endif

// ============================================================================
// Logging
// ============================================================================

static void infer_log(const InferContext *ctx, const char *fmt, ...) {
    if (!ctx || !ctx->opts.verbose) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// ============================================================================
// Memory helpers
// ============================================================================

static float* alloc_zeros(size_t n) {
    float *p = (float*)calloc(n, sizeof(float));
    if (!p) { fprintf(stderr, "[ERROR] OOM: %zu floats\n", n); exit(1); }
    return p;
}

// ============================================================================
// Weight manifest parsing
// ============================================================================

typedef struct {
    char   name[128];
    int64_t offset;
    int64_t size;
    int     shape[4];
    int     ndim;
} WeightEntry;

static bool parse_weights_manifest(
    const char   *json_path,
    WeightEntry **entries_out,
    int          *count_out)
{
    FILE *f = fopen(json_path, "r");
    if (!f) return false;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *buf = (char*)malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    // Count entries (simplified: count "name" keys)
    int count = 0;
    const char *p = buf;
    while ((p = strstr(p, "\"name\"")) != NULL) { count++; p++; }

    *entries_out = (WeightEntry*)calloc(count, sizeof(WeightEntry));
    *count_out   = count;

    // Parse each entry (simplified JSON parser)
    p = buf;
    int idx = 0;
    while (idx < count && (p = strstr(p, "\"name\"")) != NULL) {
        WeightEntry *e = &(*entries_out)[idx++];
        // name
        p = strchr(p, '"'); p++; p = strchr(p, '"'); p++;
        p = strchr(p, '"'); p++;
        int ni = 0;
        while (*p && *p != '"' && ni < 127) e->name[ni++] = *p++;
        e->name[ni] = '\0';
        // offset
        char *op = strstr(p, "\"offset\"");
        if (op) { op = strchr(op, ':'); e->offset = strtoll(op+1, NULL, 10); }
        // size
        char *sp = strstr(p, "\"nbytes\"");
        if (sp) { sp = strchr(sp, ':'); e->size = strtoll(sp+1, NULL, 10); }
        p++;
    }

    free(buf);
    return true;
}

// ============================================================================
// Model weight loading (from model_weights.bin + model_weights.json)
// ============================================================================

static bool load_non_expert_weights(InferContext *ctx, const char *model_dir) {
    char weights_path[600], manifest_path[600];
    snprintf(weights_path,  sizeof(weights_path),  "%s/model_weights.bin",  model_dir);
    snprintf(manifest_path, sizeof(manifest_path), "%s/model_weights.json", model_dir);

    infer_log(ctx, "[INFO] Loading weights from %s\n", weights_path);

    // mmap weights file
    size_t file_size = 0;
    void *mmap_handle = platform_mmap(weights_path, &file_size);
    if (!mmap_handle) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", weights_path);
        return false;
    }
    mmap_t *mm = (mmap_t*)mmap_handle;
    ctx->weights.mmap_handle = mmap_handle;
    ctx->weights.data_ptr    = mm->ptr;
    ctx->weights.data_size   = file_size;

    infer_log(ctx, "[INFO] Weights mapped: %.1f GB\n", file_size / 1e9);

    // Parse manifest
    WeightEntry *entries = NULL;
    int entry_count = 0;
    if (!parse_weights_manifest(manifest_path, &entries, &entry_count)) {
        fprintf(stderr, "[WARN] No manifest found, using heuristic layout\n");
    }

    // Allocate layer weight array
    int L = ctx->cfg.num_hidden_layers;
    ctx->weights.layers = (LayerWeights*)calloc(L, sizeof(LayerWeights));

    uint8_t *base = (uint8_t*)mm->ptr;
    const ModelConfig *c = &ctx->cfg;
    int H = c->hidden_size;
    int A = c->num_attention_heads;
    int KV = c->num_key_value_heads;
    int D = c->head_dim ? c->head_dim : H/A;
    int G = c->group_size;

    // Use manifest entries to find each tensor
    for (int ei = 0; ei < entry_count; ei++) {
        WeightEntry *e = &entries[ei];
        void *ptr = base + e->offset;

        // Match tensor name to weight slot
        // Embedding
        if (strstr(e->name, "embed_tokens.weight")) {
            ctx->weights.embed_w = (uint8_t*)ptr;
        }
        else if (strstr(e->name, "embed_tokens.scales")) {
            ctx->weights.embed_s = (uint16_t*)ptr;
        }
        else if (strstr(e->name, "embed_tokens.biases")) {
            ctx->weights.embed_b = (uint16_t*)ptr;
        }
        // Final norm
        else if (strstr(e->name, "model.norm.weight")) {
            ctx->weights.final_norm_w = (float*)ptr;
        }
        // LM head
        else if (strstr(e->name, "lm_head.weight")) {
            ctx->weights.lm_head_w = (uint8_t*)ptr;
        }
        else if (strstr(e->name, "lm_head.scales")) {
            ctx->weights.lm_head_s = (uint16_t*)ptr;
        }
        else if (strstr(e->name, "lm_head.biases")) {
            ctx->weights.lm_head_b = (uint16_t*)ptr;
        }
        else {
            // Try to match layer-specific tensors
            // Pattern: model.layers.{N}.{component}
            int layer_idx = -1;
            const char *layers_str = strstr(e->name, "layers.");
            if (layers_str) {
                layer_idx = atoi(layers_str + 7);
            }
            if (layer_idx >= 0 && layer_idx < L) {
                LayerWeights *lw = &ctx->weights.layers[layer_idx];
                if (strstr(e->name, "self_attn.q_proj.weight")) lw->q_proj_w = (uint8_t*)ptr;
                else if (strstr(e->name, "self_attn.q_proj.scales")) lw->q_proj_s = (const uint16_t*)ptr;
                else if (strstr(e->name, "self_attn.q_proj.biases")) lw->q_proj_b = (const uint16_t*)ptr;
                else if (strstr(e->name, "self_attn.k_proj.weight")) lw->k_proj_w = (uint8_t*)ptr;
                else if (strstr(e->name, "self_attn.k_proj.scales")) lw->k_proj_s = (const uint16_t*)ptr;
                else if (strstr(e->name, "self_attn.k_proj.biases")) lw->k_proj_b = (const uint16_t*)ptr;
                else if (strstr(e->name, "self_attn.v_proj.weight")) lw->v_proj_w = (uint8_t*)ptr;
                else if (strstr(e->name, "self_attn.v_proj.scales")) lw->v_proj_s = (const uint16_t*)ptr;
                else if (strstr(e->name, "self_attn.v_proj.biases")) lw->v_proj_b = (const uint16_t*)ptr;
                else if (strstr(e->name, "self_attn.o_proj.weight")) lw->o_proj_w = (uint8_t*)ptr;
                else if (strstr(e->name, "self_attn.o_proj.scales")) lw->o_proj_s = (const uint16_t*)ptr;
                else if (strstr(e->name, "self_attn.o_proj.biases")) lw->o_proj_b = (const uint16_t*)ptr;
                else if (strstr(e->name, "input_layernorm.weight"))  lw->input_norm_w = (float*)ptr;
                else if (strstr(e->name, "post_attention_layernorm")) lw->post_attn_norm_w = (float*)ptr;
                // Dense FFN (non-MoE layers)
                else if (strstr(e->name, "mlp.gate_proj.weight")) lw->gate_proj_w = (uint8_t*)ptr;
                else if (strstr(e->name, "mlp.gate_proj.scales")) lw->gate_proj_s = (const uint16_t*)ptr;
                else if (strstr(e->name, "mlp.up_proj.weight"))   lw->up_proj_w   = (uint8_t*)ptr;
                else if (strstr(e->name, "mlp.up_proj.scales"))   lw->up_proj_s   = (const uint16_t*)ptr;
                else if (strstr(e->name, "mlp.down_proj.weight")) lw->down_proj_w = (uint8_t*)ptr;
                else if (strstr(e->name, "mlp.down_proj.scales")) lw->down_proj_s = (const uint16_t*)ptr;
                // MoE router
                else if (strstr(e->name, "mlp.gate.weight"))      lw->router_w    = (uint8_t*)ptr;
                else if (strstr(e->name, "mlp.gate.scales"))       lw->router_s    = (const uint16_t*)ptr;
                // Mark full attention layers
                if (ctx->cfg.has_linear_attention) {
                    lw->is_full_attn = ((layer_idx % ctx->cfg.full_attn_interval) == 0);
                } else {
                    lw->is_full_attn = true;
                }
            }
        }
    }

    if (entries) free(entries);

    // Open expert files
    if (ctx->cfg.is_moe) {
        for (int l = 0; l < L; l++) {
            char expert_path[700];
            snprintf(expert_path, sizeof(expert_path),
                     "%s/packed_experts/layer_%02d.bin", model_dir, l);
            ctx->weights.layers[l].experts_fd = platform_open(expert_path);
            if (ctx->weights.layers[l].experts_fd != FILE_INVALID) {
                ctx->weights.layers[l].experts_file_size =
                    platform_file_size(ctx->weights.layers[l].experts_fd);
            }
        }
    }

    infer_log(ctx, "[INFO] Weight loading complete\n");
    return true;
}

// ============================================================================
// Scratch buffer allocation
// ============================================================================

static void alloc_scratch_buffers(InferContext *ctx) {
    const ModelConfig *c = &ctx->cfg;
    int H  = c->hidden_size;
    int A  = c->num_attention_heads;
    int KV = c->num_key_value_heads;
    int D  = c->head_dim ? c->head_dim : H/A;
    int NE = c->is_moe ? c->num_experts : 1;
    int FF = c->is_moe ? c->moe_intermediate_size : c->intermediate_size;

    ctx->buf_hidden     = alloc_zeros(H);
    ctx->buf_hidden2    = alloc_zeros(H);
    ctx->buf_q          = alloc_zeros(A  * D);
    ctx->buf_k          = alloc_zeros(KV * D);
    ctx->buf_v          = alloc_zeros(KV * D);
    ctx->buf_attn_out   = alloc_zeros(H);
    ctx->buf_gate       = alloc_zeros(FF);
    ctx->buf_up         = alloc_zeros(FF);
    ctx->buf_expert_out = alloc_zeros(H);
    ctx->buf_logits     = alloc_zeros(c->vocab_size);
    ctx->buf_router     = alloc_zeros(NE);

    // Expert data buffer: hold K experts' weights
    int K = c->num_experts_per_tok;
    if (K == 0) K = 4;
    ctx->buf_expert_data = (uint8_t*)malloc((size_t)K * c->expert_size_bytes + 4096);
}

// ============================================================================
// KV cache allocation
// ============================================================================

static void alloc_kv_cache(InferContext *ctx) {
    const ModelConfig *c = &ctx->cfg;
    int KV = c->num_key_value_heads;
    int D  = c->head_dim ? c->head_dim : c->hidden_size / c->num_attention_heads;
    int seq = c->max_seq_len > 0 ? c->max_seq_len : 131072;
    // Count full attention layers
    int full_attn_count = c->num_hidden_layers;
    if (c->has_linear_attention)
        full_attn_count = c->num_hidden_layers / c->full_attn_interval;

    size_t kv_size = (size_t)full_attn_count * seq * KV * D;
    ctx->kv_cache.k_cache = alloc_zeros(kv_size);
    ctx->kv_cache.v_cache = alloc_zeros(kv_size);
    ctx->kv_cache.seq_len = 0;

    infer_log(ctx, "[INFO] KV cache: %.1f GB\n", kv_size * 2 * sizeof(float) / 1e9);
}

// ============================================================================
// Main create/destroy
// ============================================================================

InferContext* infer_create(const ModelConfig *cfg, const InferOptions *opts) {
    InferContext *ctx = (InferContext*)calloc(1, sizeof(InferContext));
    if (!ctx) return NULL;

    ctx->cfg  = *cfg;
    ctx->opts = opts ? *opts : (InferOptions){0};

    // Auto-detect threads
    ctx->num_threads = ctx->opts.num_threads;
    if (ctx->num_threads <= 0) {
        ctx->num_threads = platform_cpu_count();
        if (ctx->num_threads > 8) ctx->num_threads = 8; // cap for good perf
    }

    // Detect CPU capabilities
    ctx->cpu_caps = cpu_detect();
    infer_log(ctx, "[INFO] CPU: %s, SIMD: %s, cores: %d\n",
              ctx->cpu_caps.arch_name, ctx->cpu_caps.simd_name, ctx->cpu_caps.num_cores);

    // Choose backend
    ctx->active_backend = BACKEND_CPU; // always have CPU fallback
    snprintf(ctx->backend_name, sizeof(ctx->backend_name),
             "CPU/%s (%d threads)", ctx->cpu_caps.simd_name, ctx->num_threads);

    // Allocate memory
    alloc_scratch_buffers(ctx);
    alloc_kv_cache(ctx);

    strncpy(ctx->stats.model_name, cfg->model_name, 255);
    strncpy(ctx->stats.backend_name, ctx->backend_name, 63);

    return ctx;
}

void infer_destroy(InferContext *ctx) {
    if (!ctx) return;

    // Close expert file descriptors
    if (ctx->weights.layers) {
        for (int l = 0; l < ctx->cfg.num_hidden_layers; l++) {
            if (ctx->weights.layers[l].experts_fd != FILE_INVALID)
                platform_close(ctx->weights.layers[l].experts_fd);
        }
        free(ctx->weights.layers);
    }

    // Unmap weights
    if (ctx->weights.mmap_handle)
        platform_munmap(ctx->weights.mmap_handle);

    // Free scratch buffers
    free(ctx->buf_hidden);
    free(ctx->buf_hidden2);
    free(ctx->buf_q);
    free(ctx->buf_k);
    free(ctx->buf_v);
    free(ctx->buf_attn_out);
    free(ctx->buf_gate);
    free(ctx->buf_up);
    free(ctx->buf_expert_out);
    free(ctx->buf_logits);
    free(ctx->buf_router);
    free(ctx->buf_expert_data);

    // Free KV cache
    free(ctx->kv_cache.k_cache);
    free(ctx->kv_cache.v_cache);

    free(ctx);
}

bool infer_load_weights(InferContext *ctx, const char *model_dir) {
    if (!ctx || !model_dir) return false;
    strncpy(ctx->opts.model_dir, model_dir, 511);
    return load_non_expert_weights(ctx, model_dir);
}

const char* infer_backend_name(const InferContext *ctx) {
    return ctx ? ctx->backend_name : "Unknown";
}

void infer_system_info(char *buf, int buf_len) {
    CPUCapabilities caps = cpu_detect();
    snprintf(buf, buf_len,
             "OS: %s | Arch: %s | SIMD: %s | Cores: %d",
             OS_NAME, ARCH_NAME, caps.simd_name, caps.num_cores);
}

void infer_get_stats(const InferContext *ctx, InferStats *stats) {
    if (ctx && stats) *stats = ctx->stats;
}

void infer_reset(InferContext *ctx) {
    if (!ctx) return;
    ctx->kv_cache.seq_len = 0;
    memset(ctx->buf_hidden,  0, ctx->cfg.hidden_size * sizeof(float));
    memset(ctx->buf_hidden2, 0, ctx->cfg.hidden_size * sizeof(float));
}

// ============================================================================
// Forward pass — Single transformer layer (CPU implementation)
// ============================================================================

static void forward_attention_layer(InferContext *ctx, int layer_idx, int pos) {
    const ModelConfig *c = &ctx->cfg;
    LayerWeights *lw = &ctx->weights.layers[layer_idx];
    float *h  = ctx->buf_hidden;
    float *h2 = ctx->buf_hidden2;
    int H  = c->hidden_size;
    int A  = c->num_attention_heads;
    int KV = c->num_key_value_heads;
    int D  = c->head_dim ? c->head_dim : H/A;
    int G  = c->group_size;

    // Input norm
    if (lw->input_norm_w)
        cpu_rms_norm(h, lw->input_norm_w, h2, H, c->rms_norm_eps);

    // Q, K, V projections
    if (lw->q_proj_w) {
        cpu_dequant_matvec_4bit(lw->q_proj_w, lw->q_proj_s, lw->q_proj_b,
                                h2, ctx->buf_q, A*D, H, G);
        cpu_dequant_matvec_4bit(lw->k_proj_w, lw->k_proj_s, lw->k_proj_b,
                                h2, ctx->buf_k, KV*D, H, G);
        cpu_dequant_matvec_4bit(lw->v_proj_w, lw->v_proj_s, lw->v_proj_b,
                                h2, ctx->buf_v, KV*D, H, G);
    }

    // RoPE
    int rotary_dim = (int)(D * c->partial_rotary_factor);
    if (rotary_dim == 0) rotary_dim = D;
    cpu_rope_embed(ctx->buf_q, ctx->buf_k, pos, A, KV, D, rotary_dim, c->rope_theta);

    // Store KV in cache
    // (Simplified: stores to cache, full attention uses all positions)
    int cache_offset = pos * KV * D;
    if (ctx->kv_cache.k_cache && cache_offset + KV*D <= (int)(ctx->cfg.max_seq_len * KV * D)) {
        memcpy(ctx->kv_cache.k_cache + layer_idx * ctx->cfg.max_seq_len * KV * D + cache_offset,
               ctx->buf_k, KV * D * sizeof(float));
        memcpy(ctx->kv_cache.v_cache + layer_idx * ctx->cfg.max_seq_len * KV * D + cache_offset,
               ctx->buf_v, KV * D * sizeof(float));
        ctx->kv_cache.seq_len = pos + 1;
    }

    // Scaled dot-product attention (CPU, causal)
    float scale = 1.0f / sqrtf((float)D);
    for (int head = 0; head < A; head++) {
        int kv_head = head * KV / A;  // GQA mapping
        float *q_h = ctx->buf_q + head * D;
        float *attn_scores = (float*)alloca((pos+1) * sizeof(float));

        // Q @ K^T
        for (int t = 0; t <= pos; t++) {
            float *k_t = ctx->kv_cache.k_cache + layer_idx * ctx->cfg.max_seq_len * KV * D
                         + t * KV * D + kv_head * D;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += q_h[d] * k_t[d];
            attn_scores[t] = dot * scale;
        }
        cpu_softmax(attn_scores, pos+1);

        // scores @ V
        float *out_h = ctx->buf_attn_out + head * D;
        memset(out_h, 0, D * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float *v_t = ctx->kv_cache.v_cache + layer_idx * ctx->cfg.max_seq_len * KV * D
                         + t * KV * D + kv_head * D;
            float s = attn_scores[t];
            for (int d = 0; d < D; d++) out_h[d] += s * v_t[d];
        }
    }

    // O projection
    if (lw->o_proj_w) {
        memset(h2, 0, H * sizeof(float));
        cpu_dequant_matvec_4bit(lw->o_proj_w, lw->o_proj_s, lw->o_proj_b,
                                ctx->buf_attn_out, h2, H, A*D, G);
    }

    // Residual add
    for (int i = 0; i < H; i++) h[i] += h2[i];
}

static void forward_ffn_layer(InferContext *ctx, int layer_idx) {
    const ModelConfig *c = &ctx->cfg;
    LayerWeights *lw = &ctx->weights.layers[layer_idx];
    float *h  = ctx->buf_hidden;
    float *h2 = ctx->buf_hidden2;
    int H  = c->hidden_size;
    int FF = c->intermediate_size;
    int G  = c->group_size;

    // Post-attention norm
    if (lw->post_attn_norm_w)
        cpu_rms_norm(h, lw->post_attn_norm_w, h2, H, c->rms_norm_eps);
    else
        memcpy(h2, h, H * sizeof(float));

    if (lw->gate_proj_w) {
        cpu_dequant_matvec_4bit(lw->gate_proj_w, lw->gate_proj_s, lw->gate_proj_b,
                                h2, ctx->buf_gate, FF, H, G);
        cpu_dequant_matvec_4bit(lw->up_proj_w, lw->up_proj_s, lw->up_proj_b,
                                h2, ctx->buf_up, FF, H, G);
        cpu_swiglu(ctx->buf_gate, ctx->buf_up, ctx->buf_gate, FF);
        memset(h2, 0, H * sizeof(float));
        cpu_dequant_matvec_4bit(lw->down_proj_w, lw->down_proj_s, lw->down_proj_b,
                                ctx->buf_gate, h2, H, FF, G);
    }

    // Residual add
    for (int i = 0; i < H; i++) h[i] += h2[i];
}

// ============================================================================
// MoE expert loading and forward pass
// ============================================================================

typedef struct {
    file_handle_t fd;
    void         *dst;
    size_t        size;
    int64_t       offset;
    bool          done;
} ExpertReadTask;

#ifndef OS_WINDOWS
static void* expert_read_thread(void *arg) {
    ExpertReadTask *t = (ExpertReadTask*)arg;
    ssize_t n = platform_pread(t->fd, t->dst, t->size, t->offset);
    t->done = (n == (ssize_t)t->size);
    return NULL;
}
#endif

static void forward_moe_layer(InferContext *ctx, int layer_idx) {
    const ModelConfig *c = &ctx->cfg;
    LayerWeights *lw = &ctx->weights.layers[layer_idx];
    float *h  = ctx->buf_hidden;
    float *h2 = ctx->buf_hidden2;
    int H  = c->hidden_size;
    int K  = c->num_experts_per_tok;
    int NE = c->num_experts;
    int FF = c->moe_intermediate_size;
    int G  = c->group_size;
    size_t ES = c->expert_size_bytes;

    // Post-attention norm
    if (lw->post_attn_norm_w)
        cpu_rms_norm(h, lw->post_attn_norm_w, h2, H, c->rms_norm_eps);
    else
        memcpy(h2, h, H * sizeof(float));

    // Router: compute routing scores
    if (lw->router_w) {
        memset(ctx->buf_router, 0, NE * sizeof(float));
        cpu_dequant_matvec_4bit(lw->router_w, lw->router_s, lw->router_b,
                                h2, ctx->buf_router, NE, H, G);
    }
    cpu_softmax(ctx->buf_router, NE);

    // TopK expert selection
    cpu_topk(ctx->buf_router, NE, K, ctx->expert_ids);

    double t_io_start = platform_now_ms();

    // Load K expert weights in parallel
    if (lw->experts_fd != FILE_INVALID && ES > 0) {
#ifndef OS_WINDOWS
        pthread_t threads[16];
        ExpertReadTask tasks[16];
        int actual_k = K < 16 ? K : 16;
        for (int ki = 0; ki < actual_k; ki++) {
            int eid = ctx->expert_ids[ki];
            tasks[ki].fd     = lw->experts_fd;
            tasks[ki].dst    = ctx->buf_expert_data + (size_t)ki * ES;
            tasks[ki].size   = ES;
            tasks[ki].offset = (int64_t)eid * ES;
            tasks[ki].done   = false;
            pthread_create(&threads[ki], NULL, expert_read_thread, &tasks[ki]);
        }
        for (int ki = 0; ki < actual_k; ki++)
            pthread_join(threads[ki], NULL);
#else
        for (int ki = 0; ki < K; ki++) {
            int eid = ctx->expert_ids[ki];
            platform_pread(lw->experts_fd, ctx->buf_expert_data + (size_t)ki * ES,
                           ES, (int64_t)eid * ES);
        }
#endif
    }

    double t_io_ms = platform_now_ms() - t_io_start;
    (void)t_io_ms;

    // Run each expert and accumulate
    memset(ctx->buf_expert_out, 0, H * sizeof(float));
    float router_weight_sum = 0.0f;

    for (int ki = 0; ki < K; ki++) {
        int eid = ctx->expert_ids[ki];
        float w = ctx->buf_router[eid];
        router_weight_sum += w;

        // Expert weight layout (from repack_experts.py):
        // [gate_w | gate_scales | gate_biases | up_w | up_scales | up_biases | down_w | down_s | down_b]
        const uint8_t *exp_data = ctx->buf_expert_data + (size_t)ki * ES;

        // Offsets (4-bit, group_size=64):
        // gate_w:      FF * (H/2) bytes
        // gate_scales: FF * (H/GROUP_SIZE) * 2 bytes
        // gate_biases: same
        size_t gate_w_off  = 0;
        size_t gate_w_size = (size_t)FF * (H/2);
        size_t gate_s_off  = gate_w_size;
        size_t gate_s_size = (size_t)FF * (H/G) * 2;
        size_t gate_b_off  = gate_s_off + gate_s_size;
        size_t up_w_off    = gate_b_off + gate_s_size;
        size_t up_s_off    = up_w_off + gate_w_size;
        size_t up_b_off    = up_s_off + gate_s_size;
        size_t down_w_off  = up_b_off + gate_s_size;
        size_t down_w_size = (size_t)H * (FF/2);
        size_t down_s_off  = down_w_off + down_w_size;
        size_t down_s_size = (size_t)H * (FF/G) * 2;
        size_t down_b_off  = down_s_off + down_s_size;

        if (down_b_off + down_s_size > ES) continue; // guard against bad expert size

        const uint8_t  *gate_w = exp_data + gate_w_off;
        const uint16_t *gate_s = (const uint16_t*)(exp_data + gate_s_off);
        const uint16_t *gate_b = (const uint16_t*)(exp_data + gate_b_off);
        const uint8_t  *up_w   = exp_data + up_w_off;
        const uint16_t *up_s   = (const uint16_t*)(exp_data + up_s_off);
        const uint16_t *up_b   = (const uint16_t*)(exp_data + up_b_off);
        const uint8_t  *down_w = exp_data + down_w_off;
        const uint16_t *down_s = (const uint16_t*)(exp_data + down_s_off);
        const uint16_t *down_b = (const uint16_t*)(exp_data + down_b_off);

        // gate & up projections
        cpu_dequant_matvec_4bit(gate_w, gate_s, gate_b, h2, ctx->buf_gate, FF, H, G);
        cpu_dequant_matvec_4bit(up_w,   up_s,   up_b,   h2, ctx->buf_up,   FF, H, G);

        // SwiGLU
        cpu_swiglu(ctx->buf_gate, ctx->buf_up, ctx->buf_gate, FF);

        // down projection into temp buffer then accumulate with router weight
        float *tmp_down = (float*)alloca(H * sizeof(float));
        memset(tmp_down, 0, H * sizeof(float));
        cpu_dequant_matvec_4bit(down_w, down_s, down_b, ctx->buf_gate, tmp_down, H, FF, G);

        for (int i = 0; i < H; i++) ctx->buf_expert_out[i] += w * tmp_down[i];
    }

    // Normalize by sum of router weights
    if (router_weight_sum > 1e-6f) {
        float inv = 1.0f / router_weight_sum;
        for (int i = 0; i < H; i++) ctx->buf_expert_out[i] *= inv;
    }

    // Residual add
    for (int i = 0; i < H; i++) h[i] += ctx->buf_expert_out[i];
}

// ============================================================================
// Full model forward pass for one token
// ============================================================================

static int forward_one_token(InferContext *ctx, int token_id, int pos) {
    const ModelConfig *c = &ctx->cfg;
    int H = c->hidden_size;
    int G = c->group_size;

    // Embedding lookup
    if (ctx->weights.embed_w && ctx->weights.embed_s && ctx->weights.embed_b) {
        cpu_embedding_lookup(ctx->weights.embed_w, ctx->weights.embed_s, ctx->weights.embed_b,
                             token_id, ctx->buf_hidden, H, G);
    } else {
        // Fallback: zero embedding (model not loaded yet)
        memset(ctx->buf_hidden, 0, H * sizeof(float));
    }

    // Forward through all transformer layers
    for (int l = 0; l < c->num_hidden_layers; l++) {
        // Attention
        forward_attention_layer(ctx, l, pos);

        // FFN or MoE
        if (c->is_moe) {
            forward_moe_layer(ctx, l);
        } else {
            forward_ffn_layer(ctx, l);
        }
    }

    // Final normalization
    if (ctx->weights.final_norm_w) {
        cpu_rms_norm(ctx->buf_hidden, ctx->weights.final_norm_w,
                     ctx->buf_hidden2, H, c->rms_norm_eps);
        memcpy(ctx->buf_hidden, ctx->buf_hidden2, H * sizeof(float));
    }

    // LM head projection → logits
    if (ctx->weights.lm_head_w) {
        memset(ctx->buf_logits, 0, c->vocab_size * sizeof(float));
        cpu_parallel_dequant_matvec_4bit(
            ctx->weights.lm_head_w,
            ctx->weights.lm_head_s,
            ctx->weights.lm_head_b,
            ctx->buf_hidden, ctx->buf_logits,
            c->vocab_size, H, G, ctx->num_threads);
    }

    // Sample (greedy or temperature)
    int next_token = 0;
    float max_logit = ctx->buf_logits[0];
    for (int i = 1; i < c->vocab_size; i++) {
        if (ctx->buf_logits[i] > max_logit) {
            max_logit   = ctx->buf_logits[i];
            next_token = i;
        }
    }
    return next_token;
}

// ============================================================================
// Generation loop
// ============================================================================

// Simple tokenize (character-level fallback if no BPE tokenizer loaded)
static int* simple_tokenize(InferContext *ctx, const char *text, int *out_count) {
    // Placeholder: just return ASCII codes. Real tokenizer loaded via tokenizer.h
    int len = (int)strlen(text);
    int *ids = (int*)malloc((len + 2) * sizeof(int));
    ids[0] = ctx->cfg.bos_token_id; // BOS
    int n = 1;
    for (int i = 0; i < len; i++) {
        ids[n++] = (unsigned char)text[i] % ctx->cfg.vocab_size;
    }
    *out_count = n;
    return ids;
}

int* infer_tokenize(InferContext *ctx, const char *text, int *out_count) {
    if (!ctx || !text) { *out_count = 0; return NULL; }
    return simple_tokenize(ctx, text, out_count);
}

const char* infer_detokenize(InferContext *ctx, int token_id) {
    static char buf[4] = {0};
    if (token_id >= 32 && token_id < 127) { buf[0] = (char)token_id; buf[1] = '\0'; }
    else { buf[0] = '?'; buf[1] = '\0'; }
    return buf;
}

int infer_generate(
    InferContext    *ctx,
    const char      *prompt,
    int              max_tokens,
    float            temperature,
    float            top_p,
    token_callback_t callback,
    void            *userdata)
{
    if (!ctx || !prompt) return -1;

    ctx->gen_start_ms = platform_now_ms();

    // Tokenize prompt
    int num_prompt_tokens = 0;
    int *prompt_tokens = infer_tokenize(ctx, prompt, &num_prompt_tokens);
    if (!prompt_tokens || num_prompt_tokens == 0) return -1;

    int generated = 0;
    int pos = ctx->kv_cache.seq_len;

    // Prefill (process prompt tokens)
    double prefill_start = platform_now_ms();
    for (int i = 0; i < num_prompt_tokens; i++) {
        forward_one_token(ctx, prompt_tokens[i], pos++);
    }
    double prefill_ms = platform_now_ms() - prefill_start;

    free(prompt_tokens);

    // Generation loop
    int cur_token = ctx->cfg.bos_token_id;
    for (int t = 0; t < max_tokens; t++) {
        double tok_start = platform_now_ms();
        int next_token = forward_one_token(ctx, cur_token, pos++);
        double tok_ms   = platform_now_ms() - tok_start;

        generated++;
        ctx->stats.total_tokens_generated++;

        double tps = 1000.0 / (tok_ms > 0.001 ? tok_ms : 0.001);
        ctx->stats.avg_tok_per_sec = tps;
        ctx->stats.context_length  = pos;

        InferToken it = {0};
        it.token_id   = next_token;
        it.time_ms    = platform_now_ms() - ctx->gen_start_ms;
        it.tok_per_sec = tps;
        it.is_eos     = (next_token == ctx->cfg.eos_token_id);
        if (t == 0) it.prefill_ms = prefill_ms;
        strncpy(it.token_text, infer_detokenize(ctx, next_token), 63);

        if (callback && !callback(&it, userdata)) break;
        if (it.is_eos) break;
        cur_token = next_token;
    }

    return generated;
}
