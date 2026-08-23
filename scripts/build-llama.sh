#!/usr/bin/env bash
# Build llama.cpp tuned for this workload: two small models, short prompts, tight latency.
#
# The flags below are not generic "make it fast" flags -- each one is here for a reason
# specific to how VoltageInputMcp uses llama.cpp. Read the notes before changing them.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
VENDOR="$ROOT/vendor"
SRC="$VENDOR/llama.cpp"

JOBS="${JOBS:-$(nproc)}"

# ---------------------------------------------------------------------------------------
# CUDA architecture.
#
# Must match the actual GPU. Building for the wrong arch either fails to load or silently
# runs a JIT-compiled fallback that is dramatically slower. Detected below; override with
# CUDA_ARCH=xx if detection is wrong.
#   8.6 = RTX 30-series (incl. 3050/3060 laptop)   8.9 = RTX 40-series   7.5 = RTX 20-series
# ---------------------------------------------------------------------------------------
detect_arch() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
    [[ -n "$cap" ]] && { echo "$cap"; return; }
  fi
  echo "86"
}
CUDA_ARCH="${CUDA_ARCH:-$(detect_arch)}"

for tool in git cmake; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool is required. On Arch: sudo pacman -S $tool" >&2; exit 1; }
done

USE_CUDA=ON
if ! command -v nvcc >/dev/null 2>&1; then
  echo "warning: nvcc not found -- building CPU-only. On Arch: sudo pacman -S cuda" >&2
  USE_CUDA=OFF
fi

mkdir -p "$VENDOR"
if [[ -d "$SRC/.git" ]]; then
  echo "==> updating llama.cpp"
  git -C "$SRC" pull --ff-only
else
  echo "==> cloning llama.cpp"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
fi

CACHE_ARGS=()
if command -v ccache >/dev/null 2>&1; then
  CACHE_ARGS=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
              -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
  echo "==> ccache found: rebuilds will be much faster"
fi

echo "==> configuring (CUDA=$USE_CUDA, arch=sm_$CUDA_ARCH, jobs=$JOBS)"
cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA="$USE_CUDA" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  \
  `# --------------------------------------------------------------------------------- ` \
  `# THE IMPORTANT ONE. We serve with --cache-type-k q8_0 --cache-type-v q8_0 AND       ` \
  `# -fa on, because quantised KV is what makes two models fit in 6 GB. Without         ` \
  `# FA_ALL_QUANTS, llama.cpp only compiles flash-attention kernels for a subset of KV  ` \
  `# type combinations, and a q8_0/q8_0 cache falls back to the slow path -- or refuses ` \
  `# to start. This flag adds build time and is not optional for this configuration.    ` \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  \
  `# f16 CUDA intermediates. Ampere has fast f16 tensor cores and the accuracy loss is  ` \
  `# irrelevant for 20-40 token grammar-constrained outputs.                            ` \
  -DGGML_CUDA_F16=ON \
  \
  `# Tune for this exact CPU. The CPU still does real work here: GBNF grammar           ` \
  `# evaluation runs per sampled token on the CPU, so it is on the critical path.       ` \
  -DGGML_NATIVE=ON \
  \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  "${CACHE_ARGS[@]}"

echo "==> building (FA_ALL_QUANTS makes this slower than a stock build -- 15-25 min)"
cmake --build "$SRC/build" --config Release -j "$JOBS" --target llama-server llama-cli llama-bench

BIN="$SRC/build/bin/llama-server"
if [[ -x "$BIN" ]]; then
  echo
  echo "built: $BIN"
  "$BIN" --version 2>&1 | head -3 || true
  echo
  echo "next:  ./scripts/fetch-models.sh lean && ./scripts/serve.sh lean"
  echo "then:  ./scripts/bench.sh          # measure, do not guess"
else
  echo "build finished but llama-server is missing -- check the output above" >&2
  exit 1
fi
