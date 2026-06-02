/*
 * inference_engine.h — Universal Flash-MoE inference engine API
 *
 * Drop-in cross-platform replacement for flash-moe's infer.m
 * Works on: macOS (Intel + Apple Silicon), Linux (x86_64/ARM64), Windows
 *
 * Models supported: Dense (1M–72B) and MoE (Mixtral, Qwen3.5-397B, etc.)
 */

#pragma once

#include "../platform/platform.h"
#include "../core/model_config.h"
#include "../backends/cpu_backend.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Inference context (opaque handle)
// ============================================================================

typedef struct InferContext InferContext;

// ============================================================================
// Options for creating an inference context
// ============================================================================

typedef enum {
    BACKEND_AUTO  = 0,   // Auto-detect best available
    BACKEND_CPU   = 1,   // CPU only (portable, always works)
    BACKEND_METAL = 2,   // Apple Metal (macOS ARM only)
    BACKEND_OPENCL= 3,   // OpenCL (cross-platform GPU)
    BACKEND_VULKAN= 4,   // Vulkan compute (Linux/Windows)
    BACKEND_CUDA  = 5,   // NVIDIA CUDA
} BackendType;

typedef struct {
    BackendType backend;
    int         num_threads;    // CPU threads (0 = auto-detect)
    bool        use_2bit;       // Use 2-bit quantized experts
    bool        verbose;        // Print timing/debug info
    bool        timing;         // Print per-layer timing
    int         max_k_experts;  // Override active experts (default=config value)
    int         gpu_kv_seq;     // Override GPU KV cache pre-allocation
    char        model_dir[512]; // Path to model directory
} InferOptions;

// ============================================================================
// Inference result / streaming callback
// ============================================================================

typedef struct {
    int    token_id;
    char   token_text[64];
    float  logprob;
    double time_ms;       // Time since generation start
    bool   is_eos;
    bool   is_think;
    double prefill_ms;    // Set on first token
    double tok_per_sec;   // Rolling token rate
} InferToken;

// Called once per generated token. Return false to stop.
typedef bool (*token_callback_t)(const InferToken *tok, void *userdata);

// ============================================================================
// Public API
// ============================================================================

/* Create a new inference context. Returns NULL on failure. */
InferContext* infer_create(const ModelConfig *cfg, const InferOptions *opts);

/* Free all resources */
void infer_destroy(InferContext *ctx);

/* Get backend name string */
const char* infer_backend_name(const InferContext *ctx);

/* Get CPU capabilities string (e.g. "AVX2+FMA, 8 cores") */
void infer_system_info(char *buf, int buf_len);

/* Run inference with streaming output */
int infer_generate(
    InferContext    *ctx,
    const char      *prompt,
    int              max_tokens,
    float            temperature,
    float            top_p,
    token_callback_t callback,
    void            *userdata);

/* Reset KV cache / state for new conversation */
void infer_reset(InferContext *ctx);

/* Load model weights into context (returns true on success) */
bool infer_load_weights(InferContext *ctx, const char *model_dir);

/* Tokenize text → token IDs (caller must free result) */
int* infer_tokenize(InferContext *ctx, const char *text, int *out_count);

/* Detokenize token ID → text */
const char* infer_detokenize(InferContext *ctx, int token_id);

/* Get context statistics */
typedef struct {
    int64_t  total_tokens_generated;
    double   avg_tok_per_sec;
    double   peak_tok_per_sec;
    size_t   ram_used_bytes;
    int      context_length;
    char     backend_name[64];
    char     model_name[256];
} InferStats;

void infer_get_stats(const InferContext *ctx, InferStats *stats);

// ============================================================================
// Internal implementation (compiled in inference_engine.c)
// ============================================================================

/* Weight layout for each transformer layer */
typedef struct {
    // Attention weights (all 4-bit quantized)
    uint8_t  *q_proj_w;    const uint16_t *q_proj_s, *q_proj_b;
    uint8_t  *k_proj_w;    const uint16_t *k_proj_s, *k_proj_b;
    uint8_t  *v_proj_w;    const uint16_t *v_proj_s, *v_proj_b;
    uint8_t  *o_proj_w;    const uint16_t *o_proj_s, *o_proj_b;

    // Dense FFN weights (for dense layers)
    uint8_t  *gate_proj_w; const uint16_t *gate_proj_s, *gate_proj_b;
    uint8_t  *up_proj_w;   const uint16_t *up_proj_s,   *up_proj_b;
    uint8_t  *down_proj_w; const uint16_t *down_proj_s, *down_proj_b;

    // Norms
    float    *input_norm_w;
    float    *post_attn_norm_w;

    // MoE router
    uint8_t  *router_w;    const uint16_t *router_s, *router_b;

    // Linear attention state (GatedDeltaNet)
    float    *conv_weight;
    float    *a_param;
    float    *b_param;

    // File descriptor for expert weights (packed_experts/layer_XX.bin)
    file_handle_t experts_fd;
    int64_t       experts_file_size;
    bool          is_full_attn;   // true for full attention layers
} LayerWeights;

/* Full model weight structure */
typedef struct {
    // Token embedding
    uint8_t  *embed_w;
    uint16_t *embed_s, *embed_b;

    // Final norm + LM head
    float    *final_norm_w;
    uint8_t  *lm_head_w;
    uint16_t *lm_head_s, *lm_head_b;

    // Per-layer weights
    LayerWeights *layers;

    // Backing memory (mmap or malloc)
    void     *mmap_handle;    // mmap'd memory (NULL if malloc'd)
    void     *data_ptr;       // raw data pointer
    size_t    data_size;
} ModelWeights;

/* KV cache (CPU) */
typedef struct {
    float *k_cache;   // [num_layers * max_seq * kv_heads * head_dim]
    float *v_cache;
    int    seq_len;
} KVCache;

/* Recurrent state for linear attention (GatedDeltaNet) */
typedef struct {
    float *conv_state;  // [num_layers * conv_kernel * total_key_dim]
    float *ssm_state;   // [num_layers * num_kv_heads * key_dim * value_dim]
} LinearAttnState;

/* Main inference context */
struct InferContext {
    ModelConfig      cfg;
    InferOptions     opts;
    CPUCapabilities  cpu_caps;
    BackendType      active_backend;
    char             backend_name[64];

    ModelWeights     weights;
    KVCache          kv_cache;
    LinearAttnState  lin_state;

    // Scratch buffers (reused each forward pass)
    float *buf_hidden;      // [hidden_size] current hidden state
    float *buf_hidden2;     // [hidden_size] secondary buffer
    float *buf_q;           // [num_heads * head_dim] Q projections
    float *buf_k;           // [num_kv_heads * head_dim]
    float *buf_v;           // [num_kv_heads * head_dim]
    float *buf_attn_out;    // [num_heads * head_dim]
    float *buf_gate;        // [moe_intermediate or intermediate]
    float *buf_up;          // same
    float *buf_expert_out;  // [hidden_size] accumulated expert outputs
    float *buf_logits;      // [vocab_size]
    float *buf_router;      // [num_experts] routing logits

    // Expert I/O buffers
    uint8_t *buf_expert_data;   // loaded expert weights (4x expert_size)
    int      expert_ids[16];    // selected expert indices

    // Threading
    int      num_threads;

    // Statistics
    InferStats stats;
    double     gen_start_ms;
};

#ifdef __cplusplus
}
#endif
