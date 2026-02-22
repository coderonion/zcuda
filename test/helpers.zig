/// Shared test helpers for zCUDA GPU integration tests.
///
/// Provides common utilities used across all integration and unit tests
/// that interact with the CUDA driver: PTX file loading and CUDA
/// context/stream initialization with skip-on-no-GPU semantics.
const std = @import("std");
const cuda = @import("zcuda");
const Io = std.Io;
const Dir = Io.Dir;
pub const driver = cuda.driver;

/// CUDA test environment: owns a context and provides the default stream.
pub const CudaTestEnv = struct {
    ctx: *driver.CudaContext,
    stream: *const driver.CudaStream,
};

/// Initialize a CUDA context on GPU 0. Returns `error.SkipZigTest` when
/// no GPU is present so the test runner marks the test as *skipped*.
pub fn initCuda() !CudaTestEnv {
    const ctx = driver.CudaContext.new(0) catch return error.SkipZigTest;
    const stream = ctx.defaultStream();
    return .{ .ctx = ctx, .stream = stream };
}

/// Read a pre-compiled PTX file from `zig-out/bin/kernel/<name>.ptx`.
/// Returns `error.SkipZigTest`-compatible errors when the file is
/// missing (caller should catch and skip).
///
/// Supports short test aliases (e.g. `"vector_add"`) that are automatically
/// mapped to the compiled PTX filenames produced by
/// `zig build compile-kernels -Dkernel-dir=examples/kernel/`.
fn mapKernelName(name: []const u8) []const u8 {
    const map = .{
        .{ "vector_add", "kernel_vector_add" },
        .{ "matmul", "kernel_matmul_naive" },
        .{ "tiled_matmul", "kernel_matmul_tiled" },
        .{ "histogram", "kernel_histogram" },
        .{ "grid_stride_demo", "kernel_grid_stride" },
        .{ "reduce_sum", "kernel_reduce_sum" },
        .{ "softmax", "kernel_softmax" },
        .{ "shared_mem_test", "kernel_shared_mem_demo" },
        .{ "math_test", "kernel_math_test" },
    };
    inline for (map) |entry| {
        if (std.mem.eql(u8, name, entry[0])) return entry[1];
    }
    return name; // no mapping needed — use as-is
}

pub fn readPtxFile(allocator: std.mem.Allocator, kernel_name: []const u8) ![]const u8 {
    // Map short test names → compiled PTX filenames produced by
    // `zig build compile-kernels -Dkernel-dir=examples/kernel/`
    const mapped = mapKernelName(kernel_name);

    var path_buf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "zig-out/bin/kernel/{s}.ptx", .{mapped});

    var threaded = Io.Threaded.init(allocator, .{ .environ = .empty });
    const io = threaded.io();

    const file = Dir.openFile(.cwd(), io, path, .{}) catch |err| {
        std.debug.print("PTX not found: {s} — run `zig build compile-kernels` first\n", .{path});
        return err;
    };
    defer file.close(io);

    var buf: [8192]u8 = undefined;
    var r = file.reader(io, &buf);
    const size = try r.getSize();
    return r.interface.readAlloc(allocator, size);
}
