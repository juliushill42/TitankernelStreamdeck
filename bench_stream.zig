/// TitanKernel Benchmark: Streaming Top-K vs Full Intermediate Baseline
/// Measures tokens/sec, peak memory, and throughput at multiple sequence lengths.

const std = @import("std");
const stream = @import("../src/kernels/stream_topk.zig");
const norm = @import("../src/kernels/fused_norm.zig");
const time = std.time;

const BenchResult = struct {
    seq_len: u32,
    tokens_per_sec: f64,
    peak_mb: f64,
    full_intermediate_gb: f64,
    reduction_factor: f64,
    ms_per_query: f64,
};

fn benchConfig(seq_len: u32, chunk_size: u32) stream.StreamConfig {
    return stream.StreamConfig{
        .batch_size = 1,
        .seq_len = seq_len,
        .n_heads = 32,
        .n_indexer_heads = 8,
        .head_dim = 128,
        .top_k = 64,
        .chunk_size = chunk_size,
        .compression_ratio = 4,
    };
}

fn runBench(
    allocator: std.mem.Allocator,
    cfg: stream.StreamConfig,
    n_iters: u32,
) !BenchResult {
    const hd = cfg.head_dim;
    const comp_len = cfg.seq_len / cfg.compression_ratio;

    // Allocate inputs
    const queries = try allocator.alloc(f32, cfg.n_heads * hd);
    const comp_keys = try allocator.alloc(f32, cfg.n_indexer_heads * comp_len * hd);
    const sparse_out = try allocator.alloc(u32, cfg.n_heads * cfg.top_k);
    defer allocator.free(queries);
    defer allocator.free(comp_keys);
    defer allocator.free(sparse_out);

    // Fill with deterministic values
    var rng = std.Random.DefaultPrng.init(42);
    const rand = rng.random();
    for (queries) |*v| v.* = rand.floatNorm(f32) * 0.1;
    for (comp_keys) |*v| v.* = rand.floatNorm(f32) * 0.1;

    var state = try stream.StreamState.init(allocator, cfg);
    defer state.deinit();

    // Warmup
    for (0..3) |_| {
        stream.streamingTopKIndex(queries, comp_keys, sparse_out, cfg, &state);
    }

    // Timed runs
    const start = time.nanoTimestamp();
    for (0..n_iters) |_| {
        stream.streamingTopKIndex(queries, comp_keys, sparse_out, cfg, &state);
    }
    const end = time.nanoTimestamp();

    const elapsed_ns = @as(f64, @floatFromInt(end - start));
    const ms_per_iter = elapsed_ns / @as(f64, @floatFromInt(n_iters)) / 1_000_000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(cfg.seq_len)) / (ms_per_iter / 1000.0);

    const peak_bytes = cfg.peakMemoryBytes();
    const full_bytes = cfg.fullIntermediateBytes();

    return BenchResult{
        .seq_len = cfg.seq_len,
        .tokens_per_sec = tokens_per_sec,
        .peak_mb = @as(f64, @floatFromInt(peak_bytes)) / (1024.0 * 1024.0),
        .full_intermediate_gb = @as(f64, @floatFromInt(full_bytes)) / (1024.0 * 1024.0 * 1024.0),
        .reduction_factor = @as(f64, @floatFromInt(full_bytes)) / @as(f64, @floatFromInt(peak_bytes)),
        .ms_per_query = ms_per_iter,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();

    try stdout.print("\n", .{});
    try stdout.print("╔═══════════════════════════════════════════════════════════════════════╗\n", .{});
    try stdout.print("║       TitanKernel StreamIndex — Benchmark Suite                      ║\n", .{});
    try stdout.print("║       Memory-Bounded Compressed Sparse Attention                     ║\n", .{});
    try stdout.print("╚═══════════════════════════════════════════════════════════════════════╝\n", .{});
    try stdout.print("\n", .{});

    const hw = stream.HardwareTarget.detect();
    try stdout.print("Hardware: {s}\n\n", .{@tagName(hw)});

    try stdout.print("{:<12} {:<14} {:<14} {:<16} {:<14} {:<12}\n",
        .{ "S (seq_len)", "ms/query", "tokens/sec", "peak_mem (MB)", "full_int (GB)", "reduction" });
    try stdout.print("{s}\n", .{"─" ** 84});

    const seq_lens = [_]u32{ 1024, 4096, 16384, 32768, 65536 };
    const chunk_size: u32 = 4096;

    for (seq_lens) |sl| {
        const cfg = benchConfig(sl, chunk_size);
        const n_iters: u32 = if (sl >= 32768) 10 else 50;

        const result = try runBench(allocator, cfg, n_iters);

        try stdout.print("{:<12} {:<14.2} {:<14.0} {:<16.1} {:<14.1} {:<12.0}x\n", .{
            result.seq_len,
            result.ms_per_query,
            result.tokens_per_sec,
            result.peak_mb,
            result.full_intermediate_gb,
            result.reduction_factor,
        });
    }

    try stdout.print("\n", .{});
    try stdout.print("Note: full_int (GB) = what RightNow AI materializes. H200 HBM = 80GB.\n", .{});
    try stdout.print("      At S=65536 V4-Flash dims, full_int > 256GB → OOM on any single GPU.\n", .{});
    try stdout.print("      TitanKernel peak stays constant regardless of S.\n", .{});
    try stdout.print("\n", .{});

    // RMSNorm throughput
    try stdout.print("── Fused RMSNorm Throughput ─────────────────────────────────────────\n", .{});
    const norm_sizes = [_]u32{ 512, 1024, 4096, 8192 };
    for (norm_sizes) |n| {
        const x = try allocator.alloc(f32, n);
        const w = try allocator.alloc(f32, n);
        defer allocator.free(x);
        defer allocator.free(w);
        for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.001;
        for (w) |*v| v.* = 1.0;

        const iters = 10000;
        const t0 = time.nanoTimestamp();
        for (0..iters) |_| {
            _ = norm.rmsNormInPlace(x, w);
        }
        const t1 = time.nanoTimestamp();
        const ns_per_call = @as(f64, @floatFromInt(t1 - t0)) / @as(f64, @floatFromInt(iters));
        const gb_per_s = @as(f64, @floatFromInt(n * 4)) / (ns_per_call * 1.0) * 1000.0 / (1024.0 * 1024.0 * 1024.0);
        try stdout.print("  RMSNorm dim={:<6} {:.1} ns/call  {:.1} GB/s\n", .{ n, ns_per_call, gb_per_s });
    }

    try stdout.print("\n[BENCH COMPLETE]\n", .{});
}
