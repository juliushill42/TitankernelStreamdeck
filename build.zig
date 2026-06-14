const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Static library (linked into Rust orchestrator) ────────────────────────
    const lib = b.addStaticLibrary(.{
        .name = "titankernel",
        .root_source_file = b.path("src/ffi_exports.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Enable native CPU features for SIMD auto-dispatch
    lib.root_module.cpu_features_add = std.Target.Cpu.Feature.Set.empty;
    if (target.result.cpu.arch == .x86_64) {
        const Feature = std.Target.x86.Feature;
        lib.root_module.cpu_features_add.addFeature(@intFromEnum(Feature.avx2));
    }

    b.installArtifact(lib);

    // ── Standalone test runner ────────────────────────────────────────────────
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/kernels/stream_topk.zig"),
        .target = target,
        .optimize = optimize,
    });

    const simd_tests = b.addTest(.{
        .root_source_file = b.path("src/simd/dispatch.zig"),
        .target = target,
        .optimize = optimize,
    });

    const norm_tests = b.addTest(.{
        .root_source_file = b.path("src/kernels/fused_norm.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const run_simd_tests = b.addRunArtifact(simd_tests);
    const run_norm_tests = b.addRunArtifact(norm_tests);

    const test_step = b.step("test", "Run all kernel tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_simd_tests.step);
    test_step.dependOn(&run_norm_tests.step);

    // ── Benchmark binary ──────────────────────────────────────────────────────
    const bench = b.addExecutable(.{
        .name = "titan_bench",
        .root_source_file = b.path("benchmarks/bench_stream.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench.addIncludePath(b.path("include"));
    b.installArtifact(bench);

    const bench_step = b.step("bench", "Run streaming top-k benchmark");
    const run_bench = b.addRunArtifact(bench);
    bench_step.dependOn(&run_bench.step);
}
