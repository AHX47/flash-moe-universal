#!/usr/bin/env bash
# setup.sh — Full setup: install deps, build, and prepare model tools
# Usage: ./setup.sh [--model /path/to/model]

set -e
echo "╔══════════════════════════════════════════════════════╗"
echo "║       Flash-MoE Universal — Setup Script             ║"
echo "║  macOS Intel/ARM · Linux x86/ARM · Windows           ║"
echo "╚══════════════════════════════════════════════════════╝"

OS=$(uname -s)
ARCH=$(uname -m)

# ── Install Python deps ──────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
pip install --quiet torch safetensors numpy PyQt5 huggingface_hub 2>/dev/null || \
pip3 install --quiet torch safetensors numpy PyQt5 huggingface_hub

# ── Install system build deps ────────────────────────────────
echo "[2/4] Checking build tools..."
if [ "$OS" = "Linux" ]; then
    which cmake &>/dev/null || sudo apt-get install -y cmake build-essential 2>/dev/null || \
        sudo dnf install -y cmake gcc g++ 2>/dev/null || true
    # OpenBLAS for fast CPU matmul
    sudo apt-get install -y libopenblas-dev 2>/dev/null || true
    # OpenCL
    sudo apt-get install -y ocl-icd-opencl-dev opencl-headers 2>/dev/null || true
elif [ "$OS" = "Darwin" ]; then
    which cmake &>/dev/null || brew install cmake 2>/dev/null || true
fi

# ── Build the binary ─────────────────────────────────────────
echo "[3/4] Building flash_moe binary..."
chmod +x build.sh
./build.sh

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "[4/4] ✅ Setup complete!"
echo ""
echo "Usage:"
echo "  ./build/flash_moe --info                    # System info"
echo "  ./build/flash_moe --model PATH --chat       # Interactive chat"
echo "  ./build/flash_moe --model PATH --serve 8080 # HTTP server"
echo "  python gui/app.py                           # Launch GUI"
echo ""
echo "To prepare a model:"
echo "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./my-model"
echo "  python scripts/extract_weights.py --model ./my-model --output ./my-model"
echo "  # For MoE models only:"
echo "  python scripts/repack_experts.py  --model ./my-model --output ./my-model"
