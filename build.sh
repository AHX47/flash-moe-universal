#!/usr/bin/env bash
# build.sh — Universal build script for Flash-MoE Universal
#
# Detects platform and configures CMake accordingly
# Usage:
#   ./build.sh              # Auto-detect and build
#   ./build.sh --metal      # Enable Metal (macOS ARM only)
#   ./build.sh --opencl     # Enable OpenCL
#   ./build.sh --cuda       # Enable CUDA (NVIDIA)
#   ./build.sh --cpu-only   # CPU only, no GPU

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
NPROC=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

# Parse args
USE_METAL=OFF
USE_OPENCL=OFF
USE_CUDA=OFF
USE_VULKAN=OFF

for arg in "$@"; do
    case "$arg" in
        --metal)    USE_METAL=ON ;;
        --opencl)   USE_OPENCL=ON ;;
        --cuda)     USE_CUDA=ON ;;
        --vulkan)   USE_VULKAN=ON ;;
        --cpu-only) ;;
        --clean)    rm -rf "$BUILD_DIR" ;;
    esac
done

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "╔══════════════════════════════════════════╗"
echo "║   Flash-MoE Universal Build System       ║"
echo "╚══════════════════════════════════════════╝"
echo "OS:   $OS"
echo "Arch: $ARCH"
echo "Jobs: $NPROC"
echo ""

# Auto-enable best backend
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ] && [ "$USE_METAL" = "OFF" ]; then
    echo "Detected Apple Silicon → enabling Metal backend"
    USE_METAL=ON
fi

# Check for OpenCL
if [ "$USE_OPENCL" = "OFF" ]; then
    if command -v clinfo &>/dev/null || [ -f /usr/include/CL/opencl.h ]; then
        echo "OpenCL detected → enabling OpenCL backend"
        USE_OPENCL=ON
    fi
fi

# Check for CUDA
if [ "$USE_CUDA" = "OFF" ] && command -v nvcc &>/dev/null; then
    echo "CUDA detected → enabling CUDA backend"
    USE_CUDA=ON
fi

echo ""
echo "Configuration:"
echo "  Metal:  $USE_METAL"
echo "  OpenCL: $USE_OPENCL"
echo "  CUDA:   $USE_CUDA"
echo "  Vulkan: $USE_VULKAN"
echo ""

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_METAL="$USE_METAL" \
    -DUSE_OPENCL="$USE_OPENCL" \
    -DUSE_CUDA="$USE_CUDA" \
    -DUSE_VULKAN="$USE_VULKAN" \
    -DENABLE_SIMD=ON \
    -DUSE_BLAS=ON

cmake --build . --config Release -j "$NPROC"

echo ""
echo "✅ Build successful!"
echo "   Binary: $BUILD_DIR/flash_moe"
echo ""
echo "Quick start:"
echo "  $BUILD_DIR/flash_moe --info"
echo "  $BUILD_DIR/flash_moe --model /path/to/model --prompt 'Hello'"
echo "  $BUILD_DIR/flash_moe --model /path/to/model --chat"
