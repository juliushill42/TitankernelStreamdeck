/// TitanKernel: Fused RMSNorm + Softmax
/// Fused to eliminate intermediate tensor writes — single-pass normalization
/// feeding directly into attention score softmax.
/// Hardware-agnostic, comptime SIMD.

const std = @import("std");
const builtin = @import("builtin");
const math = std.math;

pub const EPSILON: f32 = 1e-6;

/// Fused RMSNorm: normalize input in-place, return rms for reuse
pub fn rmsNormInPlace(x: []f32, weight: []const f32) f32 {
    std.debug.assert(x.len == weight.len);
    const n = x.len;

    // Compute mean square
    var ss: f32 = 0.0;
    switch (comptime detectSIMDWidth()) {
        8 => {
            const Vec8 = @Vector(8, f32);
            var i: usize = 0;
            while (i + 8 <= n) : (i += 8) {
                const v: Vec8 = x[i..][0..8].*;
                ss += @reduce(.Add, v * v);
            }
            while (i < n) : (i += 1) ss += x[i] * x[i];
        },
        16 => {
            const Vec16 = @Vector(16, f32);
            var i: usize = 0;
            while (i + 16 <= n) : (i += 16) {
                const v: Vec16 = x[i..][0..16].*;
                ss += @reduce(.Add, v * v);
            }
            while (i < n) : (i += 1) ss += x[i] * x[i];
        },
        else => {
            for (x) |v| ss += v * v;
        },
    }

    const rms = math.sqrt(ss / @as(f32, @floatFromInt(n)) + EPSILON);
    const scale = 1.0 / rms;

    // Normalize and apply weight in one pass
    for (x, weight) |*xi, wi| {
        xi.* = (xi.* * scale) * wi;
    }

    return rms;
}

/// Numerically stable softmax in-place over a slice
pub fn softmaxInPlace(x: []f32) void {
    var max_val = x[0];
    for (x[1..]) |v| if (v > max_val) { max_val = v; };

    var sum: f32 = 0.0;
    for (x) |*v| {
        v.* = math.exp(v.* - max_val);
        sum += v.*;
    }
    const inv_sum = 1.0 / sum;
    for (x) |*v| v.* *= inv_sum;
}

/// Fused temperature-scaled softmax: avoids a separate scaling pass
pub fn scaledSoftmaxInPlace(x: []f32, scale: f32) void {
    var max_val = x[0] * scale;
    for (x[1..]) |v| {
        const sv = v * scale;
        if (sv > max_val) max_val = sv;
    }

    var sum: f32 = 0.0;
    for (x) |*v| {
        v.* = math.exp(v.* * scale - max_val);
        sum += v.*;
    }
    const inv_sum = 1.0 / sum;
    for (x) |*v| v.* *= inv_sum;
}

fn detectSIMDWidth() comptime_int {
    if (builtin.cpu.arch == .x86_64) {
        if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) return 16;
        if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) return 8;
    }
    if (builtin.cpu.arch == .aarch64) return 4;
    return 1;
}

test "rmsnorm correctness" {
    const allocator = std.testing.allocator;
    const n = 32;
    const x = try allocator.alloc(f32, n);
    const w = try allocator.alloc(f32, n);
    defer allocator.free(x);
    defer allocator.free(w);

    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 + 0.1;
    for (w) |*v| v.* = 1.0; // Identity weight

    const rms = rmsNormInPlace(x, w);
    try std.testing.expect(rms > 0.0);

    // After RMSNorm with unit weights, squared mean should be ~1
    var ss: f32 = 0.0;
    for (x) |v| ss += v * v;
    const mean_sq = ss / @as(f32, @floatFromInt(n));
    try std.testing.expectApproxEqAbs(1.0, mean_sq, 0.01);
}

test "softmax sum-to-one" {
    const allocator = std.testing.allocator;
    const x = try allocator.alloc(f32, 16);
    defer allocator.free(x);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.5 - 2.0;

    softmaxInPlace(x);

    var sum: f32 = 0.0;
    for (x) |v| sum += v;
    try std.testing.expectApproxEqAbs(1.0, sum, 1e-5);
    for (x) |v| try std.testing.expect(v >= 0.0 and v <= 1.0);
}
