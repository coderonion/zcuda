/// zCUDA: NVRTC API - Safe abstraction layer.
///
/// Layer 3: High-level API for runtime compilation of CUDA C++ source
/// to PTX code. This is the recommended API.
///
/// ## Example
///
/// ```zig
/// const ptx = try nvrtc.compilePtx(allocator,
///     \\extern "C" __global__ void kernel() { }
/// );
/// defer allocator.free(ptx);
/// ```
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");

pub const NvrtcError = result.NvrtcError;

// ============================================================================
// Compile Options
// ============================================================================

/// Options for NVRTC compilation.
pub const CompileOptions = struct {
    /// Target architecture (e.g., "compute_89" for SM 8.9).
    arch: ?[]const u8 = null,
    /// Generate device debug information.
    debug: bool = false,
    /// Generate line number information.
    lineinfo: bool = false,
    /// Enable device code optimization (--dopt=on).
    dopt: bool = false,
    /// Enable flush-to-zero mode for denormals.
    ftz: bool = false,
    /// Maximum number of registers per thread.
    maxrregcount: ?u32 = null,
    /// Additional compilation options.
    extra_options: []const []const u8 = &.{},

    /// Build a list of null-terminated option strings for nvrtcCompileProgram.
    /// Helper to format a null-terminated string (allocPrintZ removed in Zig 0.16.0).
    fn fmtZ(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![:0]u8 {
        const str = try std.fmt.allocPrint(allocator, fmt, args);
        defer allocator.free(str);
        return try allocator.dupeZ(u8, str);
    }

    pub fn buildOptions(self: CompileOptions, allocator: std.mem.Allocator) !std.ArrayListUnmanaged([:0]u8) {
        var options = std.ArrayListUnmanaged([:0]u8){};
        errdefer {
            for (options.items) |opt| allocator.free(opt);
            options.deinit(allocator);
        }

        if (self.arch) |arch| {
            const opt = try fmtZ(allocator, "--gpu-architecture={s}", .{arch});
            try options.append(allocator, opt);
        }

        if (self.debug) {
            const opt = try allocator.dupeZ(u8, "--device-debug");
            try options.append(allocator, opt);
        }

        if (self.lineinfo) {
            const opt = try allocator.dupeZ(u8, "--generate-line-info");
            try options.append(allocator, opt);
        }

        if (self.dopt) {
            const opt = try allocator.dupeZ(u8, "--dopt=on");
            try options.append(allocator, opt);
        }

        if (self.ftz) {
            const opt = try allocator.dupeZ(u8, "--ftz=true");
            try options.append(allocator, opt);
        }

        if (self.maxrregcount) |count| {
            const opt = try fmtZ(allocator, "--maxrregcount={d}", .{count});
            try options.append(allocator, opt);
        }

        for (self.extra_options) |extra| {
            const opt = try allocator.dupeZ(u8, extra);
            try options.append(allocator, opt);
        }

        return options;
    }
};

// ============================================================================
// Compilation Functions
// ============================================================================

/// Compile CUDA C++ source code to PTX using default options.
///
/// Returns the PTX as an allocated byte slice. The caller must free it.
pub fn compilePtx(allocator: std.mem.Allocator, src: []const u8) ![]u8 {
    return compilePtxWithOptions(allocator, src, .{});
}

/// Compile CUDA C++ source code to PTX with custom options.
///
/// Returns the PTX as an allocated byte slice. The caller must free it.
///
/// ## Example
///
/// ```zig
/// const ptx = try compilePtxWithOptions(allocator, src, .{
///     .arch = "compute_89",
///     .opt_level = 3,
/// });
/// defer allocator.free(ptx);
/// ```
pub fn compilePtxWithOptions(
    allocator: std.mem.Allocator,
    src: []const u8,
    options: CompileOptions,
) ![]u8 {
    // Ensure source is null-terminated
    const src_z = try allocator.dupeZ(u8, src);
    defer allocator.free(src_z);

    // Create the program
    const prog = try result.createProgram(src_z.ptr, null);
    var prog_mut = prog;
    defer result.destroyProgram(&prog_mut) catch {};

    // Build compile options
    var opts = try options.buildOptions(allocator);
    defer {
        for (opts.items) |opt| allocator.free(opt);
        opts.deinit(allocator);
    }

    // Convert to pointer array
    const opts_ptrs = try allocator.alloc([*:0]const u8, opts.items.len);
    defer allocator.free(opts_ptrs);
    for (opts.items, 0..) |opt, i| {
        opts_ptrs[i] = opt.ptr;
    }

    // Compile
    result.compileProgram(prog, opts_ptrs) catch |err| {
        // On failure, try to get the compilation log for debugging
        if (result.getProgramLogSize(prog)) |log_size| {
            if (log_size > 1) {
                const log = allocator.alloc(u8, log_size) catch return err;
                defer allocator.free(log);
                result.getProgramLog(prog, log) catch return err;
                std.debug.print("NVRTC compilation error:\n{s}\n", .{log});
            }
        } else |_| {}
        return err;
    };

    // Extract PTX
    const ptx_size = try result.getPTXSize(prog);
    const ptx = try allocator.alloc(u8, ptx_size);
    errdefer allocator.free(ptx);
    try result.getPTX(prog, ptx);

    return ptx;
}

/// Get the NVRTC version.
pub fn getVersion() !result.NvrtcVersion {
    return try result.getVersion();
}

/// Compile CUDA C++ source code to CUBIN using default options.
/// CUBIN is native GPU code, faster to load than PTX.
///
/// Returns the CUBIN as an allocated byte slice. The caller must free it.
pub fn compileCubin(allocator: std.mem.Allocator, src: []const u8) ![]u8 {
    return compileCubinWithOptions(allocator, src, .{});
}

/// Compile CUDA C++ source code to CUBIN with custom options.
/// CUBIN is target-specific native code — `arch` must specify an `sm_XX` target, not `compute_XX`.
///
/// Returns the CUBIN as an allocated byte slice. The caller must free it.
pub fn compileCubinWithOptions(
    allocator: std.mem.Allocator,
    src: []const u8,
    options: CompileOptions,
) ![]u8 {
    const src_z = try allocator.dupeZ(u8, src);
    defer allocator.free(src_z);

    const prog = try result.createProgram(src_z.ptr, null);
    var prog_mut = prog;
    defer result.destroyProgram(&prog_mut) catch {};

    var opts = try options.buildOptions(allocator);
    defer {
        for (opts.items) |opt| allocator.free(opt);
        opts.deinit(allocator);
    }

    const opts_ptrs = try allocator.alloc([*:0]const u8, opts.items.len);
    defer allocator.free(opts_ptrs);
    for (opts.items, 0..) |opt, i| {
        opts_ptrs[i] = opt.ptr;
    }

    result.compileProgram(prog, opts_ptrs) catch |err| {
        if (result.getProgramLogSize(prog)) |log_size| {
            if (log_size > 1) {
                const log = allocator.alloc(u8, log_size) catch return err;
                defer allocator.free(log);
                result.getProgramLog(prog, log) catch return err;
                std.debug.print("NVRTC compilation error:\n{s}\n", .{log});
            }
        } else |_| {}
        return err;
    };

    const cubin_size = try result.getCUBINSize(prog);
    const cubin = try allocator.alloc(u8, cubin_size);
    errdefer allocator.free(cubin);
    try result.getCUBIN(prog, cubin);

    return cubin;
}

// ============================================================================
// Tests
// ============================================================================
