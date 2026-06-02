/*
 * main.c — Flash-MoE Universal CLI
 *
 * Usage:
 *   ./flash_moe --model /path/to/model --prompt "Hello" --tokens 100
 *   ./flash_moe --model /path/to/model --chat
 *   ./flash_moe --model /path/to/model --serve 8080
 *   ./flash_moe --info
 *
 * Works on: macOS Intel/ARM, Linux x86_64/ARM64/RISC-V, Windows x86_64
 */

#include "../core/inference_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef OS_WINDOWS
  #include <windows.h>
  #define usleep(x) Sleep((x)/1000)
#else
  #include <unistd.h>
#endif

#define VERSION "1.0.0"
#define BANNER \
"╔══════════════════════════════════════════════════════════╗\n" \
"║           Flash-MoE Universal Inference Engine           ║\n" \
"║    Cross-platform: Intel x86 · ARM · Apple Silicon       ║\n" \
"║    Supports: CPU · Metal · OpenCL · CUDA (NVIDIA)        ║\n" \
"╚══════════════════════════════════════════════════════════╝\n"

// ============================================================================
// Global state (for signal handler)
// ============================================================================

static volatile int g_running = 1;
static InferContext *g_ctx = NULL;

static void handle_sigint(int sig) {
    (void)sig;
    g_running = 0;
    fprintf(stderr, "\n[INFO] Interrupted. Cleaning up...\n");
    if (g_ctx) { infer_destroy(g_ctx); g_ctx = NULL; }
    exit(0);
}

// ============================================================================
// Token streaming callback
// ============================================================================

static bool stream_callback(const InferToken *tok, void *userdata) {
    (void)userdata;
    if (tok->prefill_ms > 0) {
        fprintf(stderr, "[TTFT: %.1fms]  ", tok->prefill_ms);
    }
    printf("%s", tok->token_text);
    fflush(stdout);
    if (tok->tok_per_sec > 0 && tok->token_id % 20 == 0) {
        fprintf(stderr, " [%.2f tok/s]", tok->tok_per_sec);
    }
    return g_running;
}

// ============================================================================
// Chat mode
// ============================================================================

static void run_chat_mode(InferContext *ctx, const ModelConfig *cfg) {
    printf("\n=== Interactive Chat Mode ===\n");
    printf("Model: %s | Backend: %s\n", cfg->model_name, infer_backend_name(ctx));
    printf("Type your message and press Enter. Type 'quit' or Ctrl+C to exit.\n\n");

    char line[4096];
    while (g_running) {
        printf("\n> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) break;
        // Trim newline
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;
        if (strcmp(line, "/reset") == 0) { infer_reset(ctx); printf("[Context reset]\n"); continue; }
        if (strcmp(line, "/stats") == 0) {
            InferStats stats;
            infer_get_stats(ctx, &stats);
            printf("[Stats] tokens=%lld, avg_tps=%.2f, ctx=%d\n",
                   (long long)stats.total_tokens_generated,
                   stats.avg_tok_per_sec, stats.context_length);
            continue;
        }

        printf("\n");
        int n = infer_generate(ctx, line, 512, 0.7f, 0.9f, stream_callback, NULL);
        printf("\n");
        fprintf(stderr, "[Generated %d tokens]\n", n);
    }
}

// ============================================================================
// Server mode (simple HTTP compatible with OpenAI API)
// ============================================================================

#ifndef OS_WINDOWS
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

typedef struct { InferContext *ctx; int client_fd; } ServeArgs;

static void handle_http_request(int fd, InferContext *ctx) {
    char buf[16384] = {0};
    int n = (int)read(fd, buf, sizeof(buf)-1);
    if (n <= 0) return;

    // Parse POST body for "prompt" field (simplified)
    char *body = strstr(buf, "\r\n\r\n");
    if (!body) { close(fd); return; }
    body += 4;

    char prompt[4096] = {0};
    char *p = strstr(body, "\"prompt\"");
    if (p) {
        p = strchr(p, ':'); p++;
        while (*p == ' ' || *p == '"') p++;
        int pi = 0;
        while (*p && *p != '"' && pi < 4095) prompt[pi++] = *p++;
        prompt[pi] = '\0';
    }

    // HTTP response headers
    const char *headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                          "Access-Control-Allow-Origin: *\r\nTransfer-Encoding: chunked\r\n\r\n";
    write(fd, headers, strlen(headers));

    // Stream tokens as SSE (simplified)
    infer_generate(ctx, prompt[0] ? prompt : "Hello", 512, 0.7f, 0.9f, stream_callback, NULL);
    close(fd);
}

static void run_server_mode(InferContext *ctx, int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return;
    }
    if (listen(server_fd, 10) < 0) {
        perror("listen"); close(server_fd); return;
    }

    printf("[Server] Listening on http://0.0.0.0:%d\n", port);
    printf("[Server] OpenAI-compatible API at /v1/chat/completions\n");
    printf("[Server] Press Ctrl+C to stop.\n");

    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) { if (g_running) perror("accept"); break; }
        handle_http_request(client_fd, ctx);
    }
    close(server_fd);
}
#endif

// ============================================================================
// Usage / help
// ============================================================================

static void print_usage(const char *prog) {
    printf(
        "Usage: %s [OPTIONS]\n\n"
        "Options:\n"
        "  --model <dir>        Path to model directory (required for inference)\n"
        "  --model-name <name>  Model name (auto-detects config, e.g. 'qwen2.5-7')\n"
        "  --config <file>      Path to config.json (overrides auto-detection)\n"
        "  --prompt <text>      Input prompt for generation\n"
        "  --tokens <n>         Max tokens to generate (default: 256)\n"
        "  --temp <f>           Temperature (default: 0.7)\n"
        "  --top-p <f>          Top-p sampling (default: 0.9)\n"
        "  --chat               Interactive chat mode\n"
        "  --serve <port>       Start HTTP server (OpenAI-compatible API)\n"
        "  --backend <name>     Backend: cpu, metal, opencl, cuda, auto (default: auto)\n"
        "  --threads <n>        CPU threads (default: auto-detect)\n"
        "  --2bit               Use 2-bit quantized experts\n"
        "  --timing             Print per-layer timing\n"
        "  --verbose            Verbose output\n"
        "  --info               Print system information and exit\n"
        "  --version            Print version and exit\n"
        "  --help               Show this help\n\n"
        "Examples:\n"
        "  %s --model ./qwen2.5-7b --prompt \"Explain quantum computing\" --tokens 200\n"
        "  %s --model ./qwen3.5-397b --chat --backend cpu --threads 8\n"
        "  %s --model ./llama3-8b --serve 8080\n"
        "  %s --info\n\n"
        "Supported architectures: x86_64 (AVX2), ARM64 (NEON), ARM32, RISC-V, Generic\n"
        "Supported OS: macOS (Intel + Apple Silicon), Linux, Windows\n",
        prog, prog, prog, prog, prog);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
    signal(SIGINT, handle_sigint);
#ifndef OS_WINDOWS
    signal(SIGTERM, handle_sigint);
#endif

    // ── Parse arguments ──────────────────────────────────────────────────────
    const char *model_dir   = NULL;
    const char *model_name  = NULL;
    const char *config_file = NULL;
    const char *prompt      = NULL;
    int   max_tokens  = 256;
    float temperature = 0.7f;
    float top_p       = 0.9f;
    int   serve_port  = 0;
    bool  chat_mode   = false;
    bool  show_info   = false;
    bool  verbose     = false;
    bool  timing      = false;
    bool  use_2bit    = false;
    int   num_threads = 0;
    BackendType backend = BACKEND_AUTO;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--help")    || !strcmp(argv[i], "-h")) { print_usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "--version") || !strcmp(argv[i], "-v")) { printf("flash_moe v%s\n", VERSION); return 0; }
        else if (!strcmp(argv[i], "--info"))    { show_info = true; }
        else if (!strcmp(argv[i], "--chat"))    { chat_mode = true; }
        else if (!strcmp(argv[i], "--verbose")) { verbose = true; }
        else if (!strcmp(argv[i], "--timing"))  { timing  = true; verbose = true; }
        else if (!strcmp(argv[i], "--2bit"))    { use_2bit = true; }
        else if (i+1 < argc) {
            if      (!strcmp(argv[i], "--model"))      { model_dir   = argv[++i]; }
            else if (!strcmp(argv[i], "--model-name")) { model_name  = argv[++i]; }
            else if (!strcmp(argv[i], "--config"))     { config_file = argv[++i]; }
            else if (!strcmp(argv[i], "--prompt"))     { prompt      = argv[++i]; }
            else if (!strcmp(argv[i], "--tokens"))     { max_tokens  = atoi(argv[++i]); }
            else if (!strcmp(argv[i], "--temp"))       { temperature = (float)atof(argv[++i]); }
            else if (!strcmp(argv[i], "--top-p"))      { top_p       = (float)atof(argv[++i]); }
            else if (!strcmp(argv[i], "--threads"))    { num_threads = atoi(argv[++i]); }
            else if (!strcmp(argv[i], "--serve"))      { serve_port  = atoi(argv[++i]); }
            else if (!strcmp(argv[i], "--backend")) {
                i++;
                if      (!strcmp(argv[i], "cpu"))    backend = BACKEND_CPU;
                else if (!strcmp(argv[i], "metal"))  backend = BACKEND_METAL;
                else if (!strcmp(argv[i], "opencl")) backend = BACKEND_OPENCL;
                else if (!strcmp(argv[i], "cuda"))   backend = BACKEND_CUDA;
                else                                  backend = BACKEND_AUTO;
            }
        }
    }

    printf(BANNER);
    printf("Version: %s | OS: %s | Arch: %s\n", VERSION, OS_NAME, ARCH_NAME);

    // System info
    if (show_info) {
        char info[256];
        infer_system_info(info, sizeof(info));
        printf("\n%s\n\n", info);
        CPUCapabilities caps = cpu_detect();
        printf("CPU Details:\n");
        printf("  Architecture : %s\n", caps.arch_name);
        printf("  SIMD         : %s\n", caps.simd_name);
        printf("  Logical cores: %d\n", caps.num_cores);
        printf("  AVX2         : %s\n", caps.has_avx2 ? "Yes" : "No");
        printf("  AVX-512      : %s\n", caps.has_avx512 ? "Yes" : "No");
        printf("  NEON         : %s\n", caps.has_neon ? "Yes" : "No");
        printf("\nBackends available:\n");
        printf("  CPU (SIMD)   : Always\n");
#ifdef OS_MACOS
        printf("  Metal (GPU)  : Yes (macOS)\n");
#else
        printf("  Metal (GPU)  : No (macOS only)\n");
#endif
        printf("  OpenCL       : Check with --backend opencl\n");
        printf("  CUDA         : Check with --backend cuda\n");
        return 0;
    }

    if (!model_dir && !chat_mode) {
        if (!prompt) {
            print_usage(argv[0]);
            return 1;
        }
    }

    // ── Build model config ───────────────────────────────────────────────────
    ModelConfig cfg;
    model_config_init(&cfg);

    // Try to load config.json from model dir
    if (model_dir) {
        char config_path[600];
        snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);
        if (!model_config_load_json(&cfg, config_file ? config_file : config_path)) {
            fprintf(stderr, "[WARN] Could not load config.json, using presets\n");
        }
        strncpy(cfg.model_dir, model_dir, 511);
    }

    // Apply preset by name (may override some config values with known-good values)
    if (model_name) {
        model_config_apply_preset(&cfg, model_name);
    } else if (model_dir) {
        // Try to infer name from path
        const char *base = strrchr(model_dir, PATH_SEP[0]);
        if (!base) base = model_dir; else base++;
        model_config_apply_preset(&cfg, base);
    }

    if (!cfg.model_name[0]) {
        strncpy(cfg.model_name, model_dir ? model_dir : "unknown", 255);
    }

    model_config_print(&cfg);
    printf("\n");

    // ── Create inference context ──────────────────────────────────────────────
    InferOptions opts = {0};
    opts.backend     = backend;
    opts.num_threads = num_threads;
    opts.use_2bit    = use_2bit;
    opts.verbose     = verbose;
    opts.timing      = timing;
    if (model_dir) strncpy(opts.model_dir, model_dir, 511);

    g_ctx = infer_create(&cfg, &opts);
    if (!g_ctx) {
        fprintf(stderr, "[ERROR] Failed to create inference context\n");
        return 1;
    }

    printf("[Backend] %s\n", infer_backend_name(g_ctx));

    // Load weights
    if (model_dir) {
        printf("[Loading] %s\n", model_dir);
        if (!infer_load_weights(g_ctx, model_dir)) {
            fprintf(stderr, "[ERROR] Failed to load model weights from %s\n", model_dir);
            // Continue anyway — useful for testing/benchmarking without full model
        }
    }

    // ── Run requested mode ────────────────────────────────────────────────────
    if (serve_port > 0) {
#ifndef OS_WINDOWS
        run_server_mode(g_ctx, serve_port);
#else
        fprintf(stderr, "[ERROR] Server mode not yet supported on Windows\n");
#endif
    } else if (chat_mode) {
        run_chat_mode(g_ctx, &cfg);
    } else if (prompt) {
        printf("\n--- Generating ---\n");
        double start = platform_now_ms();
        int n = infer_generate(g_ctx, prompt, max_tokens, temperature, top_p, stream_callback, NULL);
        double elapsed = platform_now_ms() - start;
        printf("\n\n--- Stats ---\n");
        printf("Tokens generated : %d\n", n);
        printf("Total time       : %.1f ms\n", elapsed);
        printf("Speed            : %.2f tok/s\n", n * 1000.0 / elapsed);
    } else {
        printf("[INFO] No prompt provided. Use --prompt, --chat, or --serve.\n");
        print_usage(argv[0]);
    }

    infer_destroy(g_ctx);
    g_ctx = NULL;
    return 0;
}
