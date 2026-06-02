
# Building Flash-MoE Universal

## Prerequisites

### All Platforms
- CMake 3.15+
- C11 compiler (GCC 9+, Clang 10+, MSVC 2019+)

### Linux
```bash
sudo apt-get install build-essential cmake libopenblas-dev
# For OpenCL support:
sudo apt-get install ocl-icd-opencl-dev opencl-headers
```
macOS

```bash
xcode-select --install
brew install cmake
```
Windows
Install Visual Studio 2022 with "Desktop development with C++"

Install CMake

(Optional) For OpenCL: Install Intel OpenCL SDK or AMD APP SDK

Build Steps
Quick Build (Recommended)
```bash
# Linux / macOS
git clone https://github.com/AHX47/flash-moe-universal
cd flash-moe-universal
./build.sh
```

# Windows
build.bat
Manual CMake Build
Linux / macOS
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```
Windows (Visual Studio)
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```
Windows (MinGW / Clang)
```bash
mkdir build
cd build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```
Build Options
Option	Default	Description
-DUSE_CPU=ON	ON	Enable CPU backend (always on)
-DUSE_METAL=OFF	OFF	Enable Metal backend (macOS ARM only)
-DUSE_OPENCL=OFF	OFF	Enable OpenCL GPU backend
-DUSE_CUDA=OFF	OFF	Enable CUDA backend (NVIDIA)
-DUSE_VULKAN=OFF	OFF	Enable Vulkan compute backend
-DENABLE_SIMD=ON	ON	Enable SIMD (AVX2/NEON)
-DUSE_BLAS=ON	ON	Use BLAS for large matrix ops
-DENABLE_GUI=ON	ON	Build PyQt5 GUI (requires Python)
Example:

```bash
cmake .. -DUSE_OPENCL=ON -DUSE_BLAS=OFF
```
Verifying the Build
```bash
./build/flash_moe --info
```
Should output system information including detected SIMD and available backends.

Troubleshooting
CMake cannot find OpenCL
```bash
Install OpenCL headers and library, or set -DOpenCL_LIBRARY=/path/to/libOpenCL.so and -DOpenCL_INCLUDE_DIR=/path/to/CL.
```

BLAS not found
```bash
Install OpenBLAS (libopenblas-dev on Debian/Ubuntu, openblas on Homebrew) or disable with -DUSE_BLAS=OFF.
```
Metal backend fails on non-Apple system
```bash
Metal is only available on macOS; disable with -DUSE_METAL=OFF.
```
Python scripts fail
```bash
Install dependencies: pip install torch safetensors numpy PyQt5
```
Build output is in different directory
```bash
The binary is always placed in build/flash_moe (or build/Release/flash_moe.exe on Windows).
```




##  API Reference


# Flash-MoE Universal API Reference

This document describes the public C API for integrating Flash-MoE into your own applications.

## Header

```c
#include "src/core/inference_engine.h"
```
Core Types
ModelConfig
Configuration structure describing the model architecture.

```c
typedef struct ModelConfig {
    char model_name[256];
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int vocab_size;
    bool is_moe;
    int num_experts;
    int num_experts_per_tok;
    // ... (see full definition in model_config.h)
} ModelConfig;
```
Initialize with model_config_init() and apply a preset or load from JSON.

InferOptions
Runtime options for inference.

```c
typedef struct InferOptions {
    BackendType backend;   // BACKEND_AUTO, BACKEND_CPU, BACKEND_METAL, etc.
    int num_threads;       // CPU threads (0 = auto-detect)
    bool use_2bit;         // Use 2-bit quantized experts
    bool verbose;          // Print debug info
    bool timing;           // Print per-layer timing
    int max_k_experts;     // Override active experts (default from config)
    char model_dir[512];   // Path to model directory
} InferOptions;
```
InferContext
Opaque handle returned by infer_create().

InferToken
Token generated during streaming.

```c
typedef struct InferToken {
    int token_id;
    char token_text[64];
    float logprob;
    double time_ms;        // Time since generation start
    bool is_eos;
    double prefill_ms;     // Set on first token
    double tok_per_sec;    // Rolling token rate
} InferToken;
Callback type: typedef bool (*token_callback_t)(const InferToken *tok, void *userdata);
```
Lifecycle Functions
infer_create
Creates a new inference context.

```c
InferContext* infer_create(const ModelConfig *cfg, const InferOptions *opts);
```
Returns NULL on failure.

infer_destroy
Frees all resources associated with the context.

```c
void infer_destroy(InferContext *ctx);```
infer_load_weights
Loads model weights (and expert files) from a directory.

```c
bool infer_load_weights(InferContext *ctx, const char *model_dir);
```
Expects model_weights.bin, model_weights.json, and (for MoE) packed_experts/ in that directory.

infer_reset
Resets KV cache and conversational state for a new session.

```c
void infer_reset(InferContext *ctx);
Generation
infer_generate
Generates text from a prompt.
```
```c
int infer_generate(
    InferContext    *ctx,
    const char      *prompt,
    int              max_tokens,
    float            temperature,
    float            top_p,
    token_callback_t callback,
    void            *userdata
);
```
Returns the number of tokens generated. The callback is called for each token (including the first). If the callback returns false, generation stops early.

Tokenizer Helpers
infer_tokenize
Tokenizes a string into token IDs. Caller must free() the result.

```c
int* infer_tokenize(InferContext *ctx, const char *text, int *out_count);
infer_detokenize
Converts a token ID back to a string (static buffer, valid until next call).
```
```c
const char* infer_detokenize(InferContext *ctx, int token_id);
Statistics
infer_get_stats
Retrieves runtime statistics.
```
```c
void infer_get_stats(const InferContext *ctx, InferStats *stats);
```
```c
typedef struct InferStats {
    int64_t  total_tokens_generated;
    double   avg_tok_per_sec;
    double   peak_tok_per_sec;
    size_t   ram_used_bytes;
    int      context_length;
    char     backend_name[64];
    char     model_name[256];
} InferStats;
Backend Information
infer_backend_name
Returns a human-readable string describing the active backend.
```
```c
const char* infer_backend_name(const InferContext *ctx);
infer_system_info
Fills a buffer with system information (OS, arch, SIMD, cores).
```
```c
void infer_system_info(char *buf, int buf_len);
```
Example
```c
#include "src/core/inference_engine.h"

int main() {
    ModelConfig cfg;
    model_config_init(&cfg);
    model_config_apply_preset(&cfg, "qwen2.5-7");

    InferOptions opts = {0};
    opts.backend = BACKEND_CPU;
    opts.num_threads = 4;

    InferContext *ctx = infer_create(&cfg, &opts);
    if (!ctx) return 1;

    infer_load_weights(ctx, "./qwen2.5-7b");

    infer_generate(ctx, "Hello, world!", 100, 0.7f, 0.9f,
        [](const InferToken *tok, void *ud) {
            printf("%s", tok->token_text);
            return !tok->is_eos;
        }, NULL);

    infer_destroy(ctx);
    return 0;
}
```
Thread Safety
The API is not thread-safe. Create separate contexts for each thread.

infer_generate must not be called concurrently on the same context.

Error Handling
Most functions return NULL, false, or a negative value on error. Check errno (POSIX) or use GetLastError() (Windows) for system-level failures.
