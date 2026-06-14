# TitanKernel StreamIndex

**Memory-Bounded Compressed Sparse Attention — Hardware-Agnostic**

State-of-the-art compressed sparse attention implementations OOM at S=65,536 on an H200.  
TitanKernel processes the same workload in **60 MB** on any hardware with a C compiler.

---

## The Problem

DeepSeek V4-Flash Compressed Sparse Attention (CSA) requires materializing a  
`[B, S, H₁, T]` FP32 score tensor before top-k selection.

At V4-Flash production dimensions:

| Parameter | Value |
|-----------|-------|
| Indexer heads H₁ | 64 |
| Compression ratio m | 4 |
| Sequence length S | 65,536 |
| Full intermediate size | **256 GB** |
| H200 HBM capacity | 80 GB |
| Result | **OOM** |

## The Solution

TitanKernel never materializes that tensor.

A chunked partition-merge streaming top-k driver processes the compressed key bank in fixed-size windows, maintaining a min-heap of `top_k` candidates per head. Peak memory is `O(chunk_size × head_dim)` — **constant with respect to sequence length**.

| Sequence Length | Full Intermediate | TitanKernel Peak | Reduction |
|-----------------|-------------------|------------------|-----------|
| 4,096 | 1.0 GB | ~60 MB | 17x |
| 16,384 | 16 GB | ~60 MB | 273x |
| 32,768 | 65 GB | ~60 MB | 1,092x |
| **65,536** | **262 GB (OOM)** | **~60 MB** | **4,369x** |
| 1,048,576 | 67 TB (OOM) | ~60 MB | 1.1M x |

---

## Hardware Support

Zero CUDA. Zero GPU required. Runs on whatever you have.

| Target | SIMD | Notes |
|--------|------|-------|
| x86_64 + AVX-512 | 16×f32 | Intel/AMD server, desktop |
| x86_64 + AVX2 | 8×f32 | Most x86 CPUs since 2013 |
| AArch64 | NEON 4×f32 | Apple Silicon, ARM servers, Raspberry Pi 5 |
| Any | Scalar | RISC-V, WASM, embedded |

SIMD target selected at **compile time** via Zig `comptime` — zero runtime branches in the hot path.

---

## Stack

```
src/kernels/stream_topk.zig   — Streaming top-k indexer + sparse attention forward
src/simd/dispatch.zig         — Comptime AVX-512 / AVX2 / NEON / scalar dot product
src/kernels/fused_norm.zig    — Fused RMSNorm + scaled softmax, single-pass
src/ffi_exports.zig           — C ABI exports (Rust FFI bridge)
rust_orch/src/main.rs         — Parallel batch scheduler, CPU topology, lifecycle mgmt
benchmarks/bench_stream.zig   — Throughput + memory benchmark
build.zig                     — Zig build: static lib, tests, benchmark
```

**Languages:** Zig 0.14, Rust 2021  
**Dependencies:** None. No BLAS. No CUDA. No Python runtime.

---

## Build

Requires: [Zig 0.14+](https://ziglang.org/download/) and [Rust 2021+](https://rustup.rs/)

```bash
git clone https://github.com/titanu-ai/titankernel
cd titankernel
chmod +x build.sh && ./build.sh
```

The build script:
1. Compiles Zig kernels → `zig-out/lib/libtitankernel.a` with `-Doptimize=ReleaseFast -Dcpu=native`
2. Runs all Zig kernel unit tests
3. Compiles Rust orchestrator, links against static lib
4. Runs streaming benchmark

### Run benchmark only

```bash
zig build bench -Doptimize=ReleaseFast
./zig-out/bin/titan_bench
```

### Run tests only

```bash
zig build test
```

---

## Benchmark Output (x86_64 + AVX2)

```
S (seq_len)  ms/query       tokens/sec     peak_mem (MB)   full_int (GB)   reduction
────────────────────────────────────────────────────────────────────────────────────
1024         0.18           5,688,888      60.1            0.1             2x
4096         0.71           5,769,014      60.1            1.0             17x
16384        2.84           5,768,845      60.1            16.4            273x
32768        5.69           5,757,292      60.1            65.5            1,092x
65536        11.38          5,759,226      60.1            262.1           4,369x
```

Peak memory flat across all sequence lengths. Throughput linear in S.

---

## Architecture

### Streaming Top-k Indexer

For each query vector, the indexer iterates over the compressed key bank in `chunk_size` row windows:

1. **Score** — dot products between query and chunk rows (SIMD-dispatched)
2. **Pack** — assemble `(score, original_key_position)` pairs
3. **Merge** — insert into per-head min-heap of capacity `top_k`; replace min when a better candidate arrives

No full score tensor ever exists in memory. Heap capacity is fixed at `top_k` entries.

### Sparse Attention Forward

After indexing, only the `top_k` selected key/value positions are loaded for each query.  
Attention complexity: `O(S × top_k)` not `O(S²)`.

### Fused RMSNorm + Softmax

Single-pass: normalizes in-place → feeds directly into temperature-scaled softmax over top-k scores.  
Eliminates one intermediate tensor write per layer.

### Rust Orchestration

- `CpuTopology` — detects logical cores, AVX-512/AVX2/NEON at runtime, selects optimal chunk size
- `StreamState` — safe Rust wrapper around Zig allocations (Send + Sync, Drop-managed)  
- `TitanScheduler` — thread pool; each worker owns its StreamState, no cross-thread mutable state

---

## Integration

The Zig kernels expose a C ABI. Link `libtitankernel.a` into any C-compatible project:

```c
// C header (include/titankernel.h)
void titan_stream_topk_index(
    const float* queries,
    const float* comp_keys,
    uint32_t* sparse_out,
    const StreamConfig* cfg,
    void* state
);
void* titan_stream_state_init(const StreamConfig* cfg);
void  titan_stream_state_free(void* state);
float titan_rmsnorm_inplace(float* x, const float* weight, uint32_t n);
void  titan_softmax_inplace(float* x, uint32_t n);
```

Works from Rust, C, C++, Python (ctypes/cffi), Go (cgo), or any language with C FFI.

---

## License

Business Source License 1.1 (BSL-1.1)  
Change Date: 2029-01-01 → Apache 2.0  
Change License: Apache License 2.0

Production use requires a commercial license until the Change Date.  
Contact: titan@titanu.ai

---

## About

Built by **Julius Cameron Hill**, Founder of Titan Universal AI, LLC.  
Atlanta, Georgia. Solo. No GPU cluster. No VC.

Titan Universal AI builds sovereign, local-first AI infrastructure — compute, network,  
model optimization, and data sovereignty — for organizations that cannot or will not  
depend on cloud providers.

titan@titanu.ai
