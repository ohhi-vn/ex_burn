#!/bin/bash
# Builds the NIF with the best GPU backend for the current platform.
#
# Usage:
#   ./build.sh              # auto-detect backend
#   ./build.sh cuda|metal|vulkan|cpu   # force a backend
set -euo pipefail

cd "$(dirname "$0")"

detect_backend() {
    # NVIDIA GPU (any OS where nvidia-smi exists and works)
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "cuda"
        return
    fi

    case "$(uname)" in
        Darwin) echo "metal" ;;
        Linux) echo "vulkan" ;;
        *) echo "cpu" ;;
    esac
}

BACKEND="${1:-$(detect_backend)}"

case "$BACKEND" in
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
        unset RUSTLER_NIF_CARGO_FEATURES
        ;;
    *)
        echo "Unknown backend: $BACKEND (expected cuda, metal, vulkan, or cpu)" >&2
        exit 1
        ;;
esac

mix compile
