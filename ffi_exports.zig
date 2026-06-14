/// TitanKernel C ABI Export Layer
/// Exposes streaming top-k and sparse attention to Rust (and any C-compatible caller).
/// This is the FFI bridge — all functions use extern "C" calling convention.

const std = @import("std");
const stream = @import("kernels/stream_topk.zig");
const norm = @import("kernels/fused_norm.zig");

const GlobalAllocator = std.heap.c_allocator;

/// Opaque state struct exposed to Rust as *mut u8
const OpaqueState = struct {
    state: stream.StreamState,
    cfg: stream.StreamConfig,
};

export fn titan_stream_state_init(cfg_c: *const stream.StreamConfig) ?*anyopaque {
    const cfg = cfg_c.*;
    const opaque_ptr = GlobalAllocator.create(OpaqueState) catch return null;
    const state = stream.StreamState.init(GlobalAllocator, cfg) catch {
        GlobalAllocator.destroy(opaque_ptr);
        return null;
    };
    opaque_ptr.* = OpaqueState{ .state = state, .cfg = cfg };
    return opaque_ptr;
}

export fn titan_stream_state_free(ptr: ?*anyopaque) void {
    if (ptr) |p| {
        const opaque: *OpaqueState = @ptrCast(@alignCast(p));
        opaque.state.deinit();
        GlobalAllocator.destroy(opaque);
    }
}

export fn titan_stream_topk_index(
    queries: [*]const f32,
    comp_keys: [*]const f32,
    sparse_out: [*]u32,
    cfg_c: *const stream.StreamConfig,
    state_ptr: ?*anyopaque,
) void {
    const cfg = cfg_c.*;
    const comp_len = cfg.seq_len / cfg.compression_ratio;

    const q_slice = queries[0 .. cfg.n_heads * cfg.head_dim];
    const ck_slice = comp_keys[0 .. cfg.n_indexer_heads * comp_len * cfg.head_dim];
    const out_slice = sparse_out[0 .. cfg.n_heads * cfg.top_k];

    if (state_ptr) |p| {
        const opaque: *OpaqueState = @ptrCast(@alignCast(p));
        stream.streamingTopKIndex(q_slice, ck_slice, out_slice, cfg, &opaque.state);
    }
}

export fn titan_sparse_attention_forward(
    queries: [*]const f32,
    keys: [*]const f32,
    values: [*]const f32,
    sparse_indices: [*]const u32,
    output: [*]f32,
    cfg_c: *const stream.StreamConfig,
) void {
    const cfg = cfg_c.*;
    const q = queries[0 .. cfg.n_heads * cfg.head_dim];
    const k = keys[0 .. cfg.n_heads * cfg.seq_len * cfg.head_dim];
    const v = values[0 .. cfg.n_heads * cfg.seq_len * cfg.head_dim];
    const idx = sparse_indices[0 .. cfg.n_heads * cfg.top_k];
    const out = output[0 .. cfg.n_heads * cfg.head_dim];

    stream.sparseAttentionForward(q, k, v, idx, out, cfg, GlobalAllocator) catch {};
}

export fn titan_rmsnorm_inplace(x: [*]f32, weight: [*]const f32, n: u32) f32 {
    const xs = x[0..n];
    const ws = weight[0..n];
    return norm.rmsNormInPlace(xs, ws);
}

export fn titan_softmax_inplace(x: [*]f32, n: u32) void {
    norm.softmaxInPlace(x[0..n]);
}

export fn titan_scaled_softmax_inplace(x: [*]f32, n: u32, scale: f32) void {
    norm.scaledSoftmaxInPlace(x[0..n], scale);
}

/// Returns peak memory bytes for a given config — callable from Rust for planning
export fn titan_peak_memory_bytes(cfg_c: *const stream.StreamConfig) u64 {
    return @intCast(cfg_c.peakMemoryBytes());
}

export fn titan_full_intermediate_bytes(cfg_c: *const stream.StreamConfig) u64 {
    return @intCast(cfg_c.fullIntermediateBytes());
}
