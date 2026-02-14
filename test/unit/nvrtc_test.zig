/// zCUDA Unit Tests: NVRTC
const std = @import("std");
const cuda = @import("zcuda");
const nvrtc = cuda.nvrtc;

test "compile simple PTX" {
    const allocator = std.testing.allocator;
    const src =
        \\extern "C" __global__ void kernel() { }
    ;
    const ptx = nvrtc.compilePtx(allocator, src) catch |err| {
        std.debug.print("Cannot compile PTX: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer allocator.free(ptx);
    try std.testing.expect(ptx.len > 0);
}

test "compile PTX with options" {
    const allocator = std.testing.allocator;
    const src =
        \\extern "C" __global__ void add(float *out, float *a, float *b, int n) {
        \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
        \\    if (i < n) out[i] = a[i] + b[i];
        \\}
    ;
    const ptx = nvrtc.compilePtxWithOptions(allocator, src, .{
        .dopt = true,
    }) catch |err| {
        std.debug.print("Cannot compile PTX with options: {}\n", .{err});
        return error.SkipZigTest;
    };
    defer allocator.free(ptx);
    try std.testing.expect(ptx.len > 0);
}

test "NVRTC version" {
    const ver = nvrtc.getVersion() catch |err| {
        std.debug.print("Cannot get NVRTC version: {}\n", .{err});
        return error.SkipZigTest;
    };
    std.debug.print("NVRTC version: {d}.{d}\n", .{ ver.major, ver.minor });
    try std.testing.expect(ver.major >= 12);
}

test "NVRTC name expression — lowered name lookup" {
    const result = @import("zcuda").nvrtc.result;

    // Create a program with a templated kernel
    const src =
        \\template<typename T>
        \\__global__ void mykernel(T *data) { data[0] = T(42); }
    ;
    const prog = result.createProgram(src, "test_kernel") catch return error.SkipZigTest;
    defer {
        var p = prog;
        result.destroyProgram(&p) catch {};
    }

    // Register the name expression before compilation
    try result.addNameExpression(prog, "mykernel<float>");

    // Compile with GPU arch
    result.compileProgram(prog, &.{"--gpu-architecture=compute_80"}) catch |err| {
        std.debug.print("NVRTC compilation failed: {}\n", .{err});
        return error.SkipZigTest;
    };

    // Get lowered name
    const lowered = try result.getLoweredName(prog, "mykernel<float>");
    const name_slice = std.mem.span(lowered);
    try std.testing.expect(name_slice.len > 0);
    std.debug.print("Lowered name: {s}\n", .{name_slice});
}

test "NVRTC supported architectures" {
    const result = @import("zcuda").nvrtc.result;

    const num_archs = result.getNumSupportedArchs() catch return error.SkipZigTest;
    try std.testing.expect(num_archs > 0);
    std.debug.print("Supported architectures: {d}\n", .{num_archs});
}
