#!/usr/bin/env bash
# TitanKernel Build Script
# Builds Zig kernels → static library, then Rust orchestrator linking against it.
# No CUDA. No GPU required.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║   TitanKernel StreamIndex — Build        ║"
echo "╚══════════════════════════════════════════╝"

# ── 1. Check toolchain ─────────────────────────────────────────────────────────
if ! command -v zig &>/dev/null; then
    echo "[ERROR] zig not found. Install from https://ziglang.org/download/"
    exit 1
fi

if ! command -v cargo &>/dev/null; then
    echo "[ERROR] cargo not found. Install from https://rustup.rs/"
    exit 1
fi

ZIG_VERSION=$(zig version)
RUST_VERSION=$(rustc --version)
echo "[OK] Zig:  $ZIG_VERSION"
echo "[OK] Rust: $RUST_VERSION"

# ── 2. Build Zig static library ───────────────────────────────────────────────
echo ""
echo "[1/4] Building Zig kernels (ReleaseFast, native CPU)..."
zig build -Doptimize=ReleaseFast -Dcpu=native 2>&1
echo "[OK]  zig-out/lib/libtitankernel.a"

# ── 3. Run Zig tests ──────────────────────────────────────────────────────────
echo ""
echo "[2/4] Running Zig kernel tests..."
zig build test 2>&1
echo "[OK]  All Zig tests passed"

# ── 4. Build Rust orchestrator ────────────────────────────────────────────────
echo ""
echo "[3/4] Building Rust orchestrator..."
cd rust_orch
cargo build --release 2>&1
echo "[OK]  target/release/titankernel"

# ── 5. Run benchmark ──────────────────────────────────────────────────────────
echo ""
echo "[4/4] Running streaming benchmark..."
cd "$SCRIPT_DIR"
zig build bench 2>&1

echo ""
echo "══════════════════════════════════════════════"
echo "  Build complete."
echo "  Binaries:"
echo "    zig-out/lib/libtitankernel.a  (Zig static lib)"
echo "    zig-out/bin/titan_bench       (Zig benchmark)"
echo "    rust_orch/target/release/titankernel (Rust orchestrator)"
echo ""
echo "  Usage:"
echo "    ./zig-out/bin/titan_bench"
echo "    ./rust_orch/target/release/titankernel"
echo "══════════════════════════════════════════════"
