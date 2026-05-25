#!/bin/bash
cd /Users/manhvu/ohhi/OSS_Lib/ex_burn

# Auto-detect the best GPU backend for this platform
detect_backend() {
    # Check for NVIDIA GPU (Linux/macOS)
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "cuda"
        return
    fi

    # Check for macOS (Metal)
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "metal"
        return
    fi

    # Check for Linux (Vulkan)
    if [[ "$(uname)" == "Linux" ]]; then
        echo "vulkan"
        return
    fi

    # Fallback: CPU only
    echo "cpu"
}

BACKEND=$(detect_backend)
echo "Detected GPU backend: $BACKEND"

case $BACKEND in
    cuda)
        echo "Building with CUDA support..."
        export RUSTLER_NIF_CARGO_FEATURES="cuda"
        ;;
    metal)
        echo "Building with Metal support..."
        export RUSTLER_NIF_CARGO_FEATURES="metal"
        ;;
    vulkan)
        echo "Building with Vulkan support..."
        export RUSTLER_NIF_CARGO_FEATURES="vulkan"
        ;;
    cpu)
        echo "Building CPU-only (NdArray)..."
        export RUSTLER_NIF_CARGO_FEATURES="cpu"
        ;;
esac

mix compile 2>&1 | head -100
