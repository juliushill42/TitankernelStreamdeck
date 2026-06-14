/// TitanKernel SIMD Dispatch Layer
/// Comptime-specialized dot product chunk scoring for AVX512/AVX2/NEON/scalar.
/// No runtime branches. No CUDA. No HBM dependency.

const std = @import("std");
const builtin = @import("builtin");

/// AVX2: 8-wide FP32 dot product over a chunk of key rows
/// Processes 8 rows per outer iteration for instruction-level parallelism
pub fn dotProductChunkAVX2(
    query: []const f32,
    keys_chunk: []const f32,
    scores_out: []f32,
    head_dim: u32,
    chunk_rows: u32,
) void {
    const Vec8f32 = @Vector(8, f32);
    var row: u32 = 0;

    // Process 8 rows at a time
    while (row + 8 <= chunk_rows) : (row += 8) {
        var acc: [8]f32 = [_]f32{0.0} ** 8;
        var d: u32 = 0;

        // Inner loop: accumulate 8 partial dot products simultaneously
        while (d + 8 <= head_dim) : (d += 8) {
            const q_vec: Vec8f32 = query[d..][0..8].*;

            // Unroll 8 rows
            inline for (0..8) |r| {
                const k_vec: Vec8f32 = keys_chunk[(row + r) * head_dim + d ..][0..8].*;
                const prod = q_vec * k_vec;
                acc[r] += @reduce(.Add, prod);
            }
        }

        // Handle tail elements
        while (d < head_dim) : (d += 1) {
            inline for (0..8) |r| {
                acc[r] += query[d] * keys_chunk[(row + r) * head_dim + d];
            }
        }

        inline for (0..8) |r| {
            scores_out[row + r] = acc[r];
        }
    }

    // Handle remaining rows with scalar
    while (row < chunk_rows) : (row += 1) {
        var acc: f32 = 0.0;
        var d: u32 = 0;
        while (d < head_dim) : (d += 1) {
            acc += query[d] * keys_chunk[row * head_dim + d];
        }
        scores_out[row] = acc;
    }
}

/// AVX512: 16-wide FP32 dot product, processes 16 rows per outer iteration
pub fn dotProductChunkAVX512(
    query: []const f32,
    keys_chunk: []const f32,
    scores_out: []f32,
    head_dim: u32,
    chunk_rows: u32,
) void {
    const Vec16f32 = @Vector(16, f32);
    var row: u32 = 0;

    while (row + 16 <= chunk_rows) : (row += 16) {
        var acc: [16]f32 = [_]f32{0.0} ** 16;
        var d: u32 = 0;

        while (d + 16 <= head_dim) : (d += 16) {
            const q_vec: Vec16f32 = query[d..][0..16].*;
            inline for (0..16) |r| {
                const k_vec: Vec16f32 = keys_chunk[(row + r) * head_dim + d ..][0..16].*;
                acc[r] += @reduce(.Add, q_vec * k_vec);
            }
        }
        // Tail
        while (d < head_dim) : (d += 1) {
            inline for (0..16) |r| {
                acc[r] += query[d] * keys_chunk[(row + r) * head_dim + d];
            }
        }
        inline for (0..16) |r| {
            scores_out[row + r] = acc[r];
        }
    }

    // Fall through to AVX2 for remainder (common sub-chunk sizes)
    if (row < chunk_rows) {
        dotProductChunkAVX2(
            query,
            keys_chunk[row * head_dim ..],
            scores_out[row..],
            head_dim,
            chunk_rows - row,
        );
    }
}

/// NEON (ARM): 4-wide FP32, unrolled 4 rows
pub fn dotProductChunkNEON(
    query: []const f32,
    keys_chunk: []const f32,
    scores_out: []f32,
    head_dim: u32,
    chunk_rows: u32,
) void {
    const Vec4f32 = @Vector(4, f32);
    var row: u32 = 0;

    while (row + 4 <= chunk_rows) : (row += 4) {
        var acc: [4]f32 = [_]f32{0.0} ** 4;
        var d: u32 = 0;

        while (d + 4 <= head_dim) : (d += 4) {
            const q_vec: Vec4f32 = query[d..][0..4].*;
            inline for (0..4) |r| {
                const k_vec: Vec4f32 = keys_chunk[(row + r) * head_dim + d ..][0..4].*;
                acc[r] += @reduce(.Add, q_vec * k_vec);
            }
        }
        while (d < head_dim) : (d += 1) {
            inline for (0..4) |r| {
                acc[r] += query[d] * keys_chunk[(row + r) * head_dim + d];
            }
        }
        inline for (0..4) |r| {
            scores_out[row + r] = acc[r];
        }
    }

    while (row < chunk_rows) : (row += 1) {
        var acc: f32 = 0.0;
        var d: u32 = 0;
        while (d < head_dim) : (d += 1) {
            acc += query[d] * keys_chunk[row * head_dim + d];
        }
        scores_out[row] = acc;
    }
}

test "avx2 dot product correctness" {
    const allocator = std.testing.allocator;
    const head_dim: u32 = 16;
    const chunk_rows: u32 = 8;

    const query = try allocator.alloc(f32, head_dim);
    const keys = try allocator.alloc(f32, chunk_rows * head_dim);
    const scores_avx2 = try allocator.alloc(f32, chunk_rows);
    const scores_ref = try allocator.alloc(f32, chunk_rows);
    defer allocator.free(query);
    defer allocator.free(keys);
    defer allocator.free(scores_avx2);
    defer allocator.free(scores_ref);

    for (query, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;
    for (keys, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % head_dim)) * 0.05;

    dotProductChunkAVX2(query, keys, scores_avx2, head_dim, chunk_rows);

    // Reference scalar
    for (0..chunk_rows) |r| {
        var acc: f32 = 0.0;
        for (0..head_dim) |d| acc += query[d] * keys[r * head_dim + d];
        scores_ref[r] = acc;
    }

    for (0..chunk_rows) |i| {
        try std.testing.expectApproxEqAbs(scores_ref[i], scores_avx2[i], 1e-4);
    }
}
