/*
 * model_config.h — Dynamic model configuration for Flash-MoE Universal
 *
 * Supports any model from 1M to 397B+ parameters:
 *   Dense models:  Qwen2.5-0.5B/1.5B/3B/7B/14B/32B/72B
 *                  Llama-3.1 8B/70B, Phi-3/4, Mistral 7B/24B, Gemma 2B/7B/9B/27B
 *   MoE models:    Qwen3.5-397B-A17B, Mixtral 8x7B, Deepseek-V2
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Model Architecture Type
// ============================================================================

typedef enum {
    ARCH_UNKNOWN    = 0,
    ARCH_QWEN2      = 1,   // Qwen2/Qwen2.5 dense
    ARCH_QWEN3_MOE  = 2,   // Qwen3.5 MoE (flash-moe original)
    ARCH_LLAMA      = 3,   // Llama-2/3
    ARCH_MISTRAL    = 4,   // Mistral/Mixtral
    ARCH_PHI        = 5,   // Phi-3/4
    ARCH_GEMMA      = 6,   // Gemma 2B/7B
    ARCH_DEEPSEEK   = 7,   // DeepSeek-V2/V3
    ARCH_GENERIC    = 99,  // Generic transformer
} ModelArch;

// ============================================================================
// Attention Type
// ============================================================================

typedef enum {
    ATTN_STANDARD   = 0,   // Standard QKV dot-product attention
    ATTN_GQA        = 1,   // Grouped Query Attention
    ATTN_MLA        = 2,   // Multi-head Latent Attention (DeepSeek)
    ATTN_LINEAR     = 3,   // GatedDeltaNet linear attention
} AttentionType;

// ============================================================================
// Model Configuration (matches config.json fields from HuggingFace)
// ============================================================================

typedef struct ModelConfig {
    // ── Identity ────────────────────────────────────────────────────────────
    char     model_name[256];
    char     model_type[64];         // "qwen2", "llama", "mistral", etc.
    ModelArch arch;

    // ── Core dimensions ─────────────────────────────────────────────────────
    int      hidden_size;            // 4096
    int      num_hidden_layers;      // 60
    int      num_attention_heads;    // 32
    int      num_key_value_heads;    // 2 (GQA)
    int      head_dim;               // 256 (or hidden_size/num_heads)
    int      intermediate_size;      // FFN hidden size (dense layers)
    int      vocab_size;             // 248320

    // ── RoPE ────────────────────────────────────────────────────────────────
    float    rope_theta;             // 10000000.0
    float    partial_rotary_factor;  // 0.25 (Qwen3.5) or 1.0 (most models)
    int      max_position_embeddings;// 131072

    // ── Normalization ────────────────────────────────────────────────────────
    float    rms_norm_eps;           // 1e-6
    bool     use_rms_norm;           // true for most recent models
    float    layer_norm_eps;         // 1e-5 (for BERT-style)

    // ── MoE (Mixture of Experts) ─────────────────────────────────────────────
    bool     is_moe;                 // true if model uses MoE
    int      num_experts;            // 512 (Qwen3.5), 8 (Mixtral)
    int      num_experts_per_tok;    // 4 (active experts per token)
    int      moe_intermediate_size;  // 1024
    bool     has_shared_expert;      // true for Qwen3.5
    int      shared_expert_intermediate_size; // 1024

    // ── Hybrid attention (Qwen3.5 specific) ──────────────────────────────────
    bool     has_linear_attention;    // GatedDeltaNet layers
    int      full_attn_interval;      // every N layers uses full attention (4)
    int      linear_num_v_heads;      // 64
    int      linear_num_k_heads;      // 16
    int      linear_key_head_dim;     // 128
    int      linear_value_head_dim;   // 128
    int      conv_kernel_size;        // 4

    // ── Quantization ─────────────────────────────────────────────────────────
    int      bits;                    // 4 or 2 (quantization bits)
    int      group_size;              // 64 (quantization group size)
    bool     has_2bit_experts;        // use 2-bit expert weights

    // ── Context / KV cache ───────────────────────────────────────────────────
    int      max_seq_len;             // 1048576 (1M)
    int      gpu_kv_seq;              // 8192 (pre-allocated GPU KV buffer)

    // ── Special tokens ───────────────────────────────────────────────────────
    int      bos_token_id;
    int      eos_token_id;
    int      pad_token_id;
    int      think_start_token;
    int      think_end_token;

    // ── File paths ───────────────────────────────────────────────────────────
    char     model_dir[512];
    char     weights_file[512];
    char     tokenizer_file[512];
    char     experts_dir[512];

    // ── Derived / computed ───────────────────────────────────────────────────
    int64_t  total_params;           // approximate total parameter count
    int64_t  active_params;          // approximate active params per token
    size_t   expert_size_bytes;      // per-expert blob size on disk
    size_t   non_expert_size_bytes;  // non-expert weights size
} ModelConfig;

// ============================================================================
// Known model presets (populated automatically by name detection)
// ============================================================================

static const struct {
    const char *name_fragment;
    ModelArch   arch;
    int         hidden_size;
    int         num_hidden_layers;
    int         num_attention_heads;
    int         num_key_value_heads;
    int         intermediate_size;
    int         vocab_size;
    float       rope_theta;
    bool        is_moe;
    int         num_experts;
    int         num_experts_per_tok;
    int         moe_intermediate_size;
} MODEL_PRESETS[] = {
    // name,          arch,         hid,  layers, h,  kv,  ffn,    vocab,   rope_theta,    moe, ne, nep, moe_ff
    {"qwen3.5-397",   ARCH_QWEN3_MOE, 4096, 60, 32, 2, 0,      248320, 1e7f,  true,  512, 4,   1024},
    {"qwen2.5-72",    ARCH_QWEN2,    8192, 80, 64, 8, 29568,   152064, 1e6f,  false, 0,   0,   0},
    {"qwen2.5-32",    ARCH_QWEN2,    5120, 64, 40, 8, 27648,   152064, 1e6f,  false, 0,   0,   0},
    {"qwen2.5-14",    ARCH_QWEN2,    5120, 48, 40, 8, 13824,   152064, 1e6f,  false, 0,   0,   0},
    {"qwen2.5-7",     ARCH_QWEN2,    3584, 28, 28, 4, 18944,   152064, 1e6f,  false, 0,   0,   0},
    {"qwen2.5-3",     ARCH_QWEN2,    2048, 36, 16, 2, 11008,   151936, 5e5f,  false, 0,   0,   0},
    {"qwen2.5-1.5",   ARCH_QWEN2,    1536, 28, 12, 2, 8960,    151936, 1e6f,  false, 0,   0,   0},
    {"qwen2.5-0.5",   ARCH_QWEN2,    896,  24, 14, 2, 4864,    151936, 1e6f,  false, 0,   0,   0},
    {"llama-3.1-70",  ARCH_LLAMA,    8192, 80, 64, 8, 28672,   128256, 5e5f,  false, 0,   0,   0},
    {"llama-3.1-8",   ARCH_LLAMA,    4096, 32, 32, 8, 14336,   128256, 5e5f,  false, 0,   0,   0},
    {"llama-3.2-3",   ARCH_LLAMA,    3072, 28, 24, 8, 8192,    128256, 5e5f,  false, 0,   0,   0},
    {"llama-3.2-1",   ARCH_LLAMA,    2048, 16, 32, 8, 8192,    128256, 5e5f,  false, 0,   0,   0},
    {"mistral-7",     ARCH_MISTRAL,  4096, 32, 32, 8, 14336,   32000,  1e6f,  false, 0,   0,   0},
    {"mixtral-8x7",   ARCH_MISTRAL,  4096, 32, 32, 8, 14336,   32000,  1e6f,  true,  8,   2,   14336},
    {"phi-4",         ARCH_PHI,      5120, 40, 40, 10,17920,   100352, 2.5e5f,false, 0,   0,   0},
    {"phi-3-mini",    ARCH_PHI,      3072, 32, 32, 32,8192,    32064,  1e4f,  false, 0,   0,   0},
    {"gemma-2-27",    ARCH_GEMMA,    4608, 46, 32, 16,36864,   256000, 1e4f,  false, 0,   0,   0},
    {"gemma-2-9",     ARCH_GEMMA,    3584, 42, 16, 8, 14336,   256000, 1e4f,  false, 0,   0,   0},
    {"gemma-2-2",     ARCH_GEMMA,    2304, 26, 8,  4, 9216,    256000, 1e4f,  false, 0,   0,   0},
    {NULL,            ARCH_GENERIC,  512,  6,  8,  8, 2048,    32000,  1e4f,  false, 0,   0,   0},
};

// ============================================================================
// Config API
// ============================================================================

/* Initialize a config with safe defaults */
static inline void model_config_init(ModelConfig *cfg) {
    memset(cfg, 0, sizeof(ModelConfig));
    cfg->hidden_size              = 4096;
    cfg->num_hidden_layers        = 32;
    cfg->num_attention_heads      = 32;
    cfg->num_key_value_heads      = 32;
    cfg->intermediate_size        = 11008;
    cfg->vocab_size               = 32000;
    cfg->rope_theta               = 10000.0f;
    cfg->partial_rotary_factor    = 1.0f;
    cfg->rms_norm_eps             = 1e-6f;
    cfg->use_rms_norm             = true;
    cfg->max_position_embeddings  = 131072;
    cfg->bits                     = 4;
    cfg->group_size               = 64;
    cfg->max_seq_len              = 131072;
    cfg->gpu_kv_seq               = 4096;
    cfg->bos_token_id             = 1;
    cfg->eos_token_id             = 2;
    cfg->pad_token_id             = 0;
    cfg->num_experts_per_tok      = 4;
    cfg->arch                     = ARCH_GENERIC;
}

/* Apply a named preset (case-insensitive substring match) */
static inline bool model_config_apply_preset(ModelConfig *cfg, const char *model_name) {
    if (!model_name) return false;
    // lowercase copy
    char lower[256] = {0};
    for (int i = 0; i < 255 && model_name[i]; i++)
        lower[i] = (char)(model_name[i] >= 'A' && model_name[i] <= 'Z'
                          ? model_name[i] + 32 : model_name[i]);

    for (int i = 0; MODEL_PRESETS[i].name_fragment != NULL; i++) {
        if (strstr(lower, MODEL_PRESETS[i].name_fragment)) {
            cfg->arch                   = MODEL_PRESETS[i].arch;
            cfg->hidden_size            = MODEL_PRESETS[i].hidden_size;
            cfg->num_hidden_layers      = MODEL_PRESETS[i].num_hidden_layers;
            cfg->num_attention_heads    = MODEL_PRESETS[i].num_attention_heads;
            cfg->num_key_value_heads    = MODEL_PRESETS[i].num_key_value_heads;
            cfg->intermediate_size      = MODEL_PRESETS[i].intermediate_size;
            cfg->vocab_size             = MODEL_PRESETS[i].vocab_size;
            cfg->rope_theta             = MODEL_PRESETS[i].rope_theta;
            cfg->is_moe                 = MODEL_PRESETS[i].is_moe;
            cfg->num_experts            = MODEL_PRESETS[i].num_experts;
            cfg->num_experts_per_tok    = MODEL_PRESETS[i].num_experts_per_tok;
            cfg->moe_intermediate_size  = MODEL_PRESETS[i].moe_intermediate_size;

            // Qwen3.5 hybrid attention
            if (cfg->arch == ARCH_QWEN3_MOE) {
                cfg->has_linear_attention      = true;
                cfg->full_attn_interval        = 4;
                cfg->head_dim                  = 256;
                cfg->partial_rotary_factor     = 0.25f;
                cfg->linear_num_v_heads        = 64;
                cfg->linear_num_k_heads        = 16;
                cfg->linear_key_head_dim       = 128;
                cfg->linear_value_head_dim     = 128;
                cfg->conv_kernel_size          = 4;
                cfg->has_shared_expert         = true;
                cfg->shared_expert_intermediate_size = 1024;
                cfg->expert_size_bytes         = 7077888;
            }
            strncpy(cfg->model_name, model_name, 255);
            return true;
        }
    }
    return false;
}

/* Load config from a JSON file (config.json from HuggingFace) */
static inline bool model_config_load_json(ModelConfig *cfg, const char *json_path) {
    FILE *f = fopen(json_path, "r");
    if (!f) return false;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return false; }
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    // Simple key-value JSON parser (no full JSON library needed)
    #define JSON_GET_INT(key, field) do { \
        char *p = strstr(buf, "\"" key "\""); \
        if (p) { p = strchr(p, ':'); if (p) cfg->field = atoi(p+1); } \
    } while(0)

    #define JSON_GET_FLOAT(key, field) do { \
        char *p = strstr(buf, "\"" key "\""); \
        if (p) { p = strchr(p, ':'); if (p) cfg->field = (float)atof(p+1); } \
    } while(0)

    #define JSON_GET_BOOL(key, field) do { \
        char *p = strstr(buf, "\"" key "\""); \
        if (p) { p = strchr(p, ':'); if (p) { \
            while (*p == ' ' || *p == ':') p++; \
            cfg->field = (strncmp(p, "true", 4) == 0); \
        }} \
    } while(0)

    JSON_GET_INT("hidden_size",                     hidden_size);
    JSON_GET_INT("num_hidden_layers",               num_hidden_layers);
    JSON_GET_INT("num_attention_heads",             num_attention_heads);
    JSON_GET_INT("num_key_value_heads",             num_key_value_heads);
    JSON_GET_INT("intermediate_size",               intermediate_size);
    JSON_GET_INT("vocab_size",                      vocab_size);
    JSON_GET_INT("num_experts",                     num_experts);
    JSON_GET_INT("num_experts_per_tok",             num_experts_per_tok);
    JSON_GET_INT("moe_intermediate_size",           moe_intermediate_size);
    JSON_GET_INT("shared_expert_intermediate_size", shared_expert_intermediate_size);
    JSON_GET_INT("max_position_embeddings",         max_position_embeddings);
    JSON_GET_INT("bos_token_id",                    bos_token_id);
    JSON_GET_INT("eos_token_id",                    eos_token_id);
    JSON_GET_FLOAT("rope_theta",                    rope_theta);
    JSON_GET_FLOAT("partial_rotary_factor",         partial_rotary_factor);
    JSON_GET_FLOAT("rms_norm_eps",                  rms_norm_eps);
    JSON_GET_BOOL("use_sliding_window",             has_linear_attention);

    // Detect model_type string
    char *mt = strstr(buf, "\"model_type\"");
    if (mt) {
        mt = strchr(mt, ':');
        if (mt) {
            mt++; while (*mt == ' ' || *mt == '"') mt++;
            int i = 0;
            while (mt[i] && mt[i] != '"' && i < 63) { cfg->model_type[i] = mt[i]; i++; }
            cfg->model_type[i] = '\0';
        }
    }

    // Detect arch from model_type
    if (strstr(cfg->model_type, "qwen3")) cfg->arch = cfg->is_moe ? ARCH_QWEN3_MOE : ARCH_QWEN2;
    else if (strstr(cfg->model_type, "qwen")) cfg->arch = ARCH_QWEN2;
    else if (strstr(cfg->model_type, "llama")) cfg->arch = ARCH_LLAMA;
    else if (strstr(cfg->model_type, "mistral") || strstr(cfg->model_type, "mixtral")) cfg->arch = ARCH_MISTRAL;
    else if (strstr(cfg->model_type, "phi")) cfg->arch = ARCH_PHI;
    else if (strstr(cfg->model_type, "gemma")) cfg->arch = ARCH_GEMMA;

    cfg->is_moe = (cfg->num_experts > 1);
    if (!cfg->head_dim)
        cfg->head_dim = cfg->hidden_size / cfg->num_attention_heads;

    free(buf);
    return true;
}

/* Print config summary */
static inline void model_config_print(const ModelConfig *cfg) {
    printf("=== Model Config: %s ===\n", cfg->model_name);
    printf("  Architecture : %s (arch=%d)\n", cfg->model_type, cfg->arch);
    printf("  Layers       : %d\n",  cfg->num_hidden_layers);
    printf("  Hidden size  : %d\n",  cfg->hidden_size);
    printf("  Attention    : %d heads (%d kv), head_dim=%d\n",
           cfg->num_attention_heads, cfg->num_key_value_heads, cfg->head_dim);
    printf("  Vocab        : %d\n",  cfg->vocab_size);
    if (cfg->is_moe) {
        printf("  MoE          : %d experts, K=%d active per token\n",
               cfg->num_experts, cfg->num_experts_per_tok);
        printf("  MoE FFN size : %d\n",  cfg->moe_intermediate_size);
        if (cfg->has_shared_expert)
            printf("  Shared expert FFN : %d\n", cfg->shared_expert_intermediate_size);
    } else {
        printf("  FFN size     : %d\n",  cfg->intermediate_size);
    }
    printf("  Quantization : %d-bit, group_size=%d\n", cfg->bits, cfg->group_size);
    printf("  Context      : %d max tokens\n", cfg->max_seq_len);
}

#ifdef __cplusplus
}
#endif
