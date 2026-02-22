/// bridge_gen.zig — Comptime bridge generator for zcuda kernel modules (Way 5)
///
/// Part of the public zcuda API. Generates type-safe bridge interfaces for
/// GPU kernels compiled from Zig source to PTX.
///
/// ## For zcuda users
///
/// In your **build.zig**, create a bridge module for each kernel:
///
/// ```zig
/// const zcuda_bridge = b.dependency("zcuda", .{}).module("zcuda_bridge");
///
/// // Generate a one-liner bridge module via addWriteFiles
/// const wf = b.addWriteFiles();
/// const bridge_src = wf.add("kernel_my_kernel.zig",
///     \\pub usingnamespace @import("zcuda_bridge").init(.{
///     \\    .name = "my_kernel",
///     \\    .ptx_path = "zig-out/bin/kernel/my_kernel.ptx",
///     \\    .fn_names = &.{ "myFunc", "myOtherFunc" },
///     \\});
/// );
///
/// // Create the importable module
/// const my_kernel_mod = b.addModule("kernel_my_kernel", .{
///     .root_source_file = bridge_src,
/// });
/// my_kernel_mod.addImport("zcuda_bridge", zcuda_bridge);
/// my_kernel_mod.addImport("zcuda", zcuda_mod);
///
/// // Wire it into your executable
/// exe.root_module.addImport("kernel_my_kernel", my_kernel_mod);
/// ```
///
/// In your **application code**:
///
/// ```zig
/// const kernel = @import("kernel_my_kernel");
///
/// const module = try kernel.load(ctx, allocator);
/// defer module.deinit();
/// const func = try kernel.getFunction(module, .myFunc);
/// try stream.launch(func, grid, block, .{ d_ptr, n });
/// ```
///
/// ## Design
///
/// - **Comptime type safety**: `Fn` enum is built at comptime from your function
///   names. Typos in `.getFunction(mod, .wrongName)` cause a build error.
/// - **Generic**: works with any PTX path — not tied to zcuda's own examples.
/// - **Single source of truth**: all bridge logic lives here, not in N generated files.
/// - **IDE-friendly**: real Zig code → autocomplete, Ctrl+Click, hover docs all work.
const std = @import("std");

/// Configuration for a kernel bridge module.
pub const Config = struct {
    /// Kernel name (used for logging and identification).
    name: []const u8,

    /// Path to the compiled PTX file (used in disk-loading mode).
    /// Can be absolute or relative to the working directory.
    /// Example: "zig-out/bin/kernel/my_kernel.ptx"
    ptx_path: []const u8,

    /// Exported function names in the kernel (used to build the Fn enum).
    fn_names: []const []const u8,

    /// Optional: path to the kernel source file (for documentation).
    source_path: ?[]const u8 = null,

    /// Optional: compile-time embedded PTX data.
    /// When non-null, `load()` uses this data directly (zero file I/O).
    /// Set via `@embedFile("ptx_data")` in the generated bridge source
    /// when building with `--embed-ptx`.
    ptx_data: ?[]const u8 = null,
};

/// Generate a kernel bridge module type with type-safe function access.
///
/// Returns a struct with:
///   - `Fn`: enum of exported function names (compile-time checked)
///   - `load()`: read PTX file and load as CudaModule
///   - `getFunction()`: get CudaFunction handle by enum name
///   - `name`, `ptx_path`, `source_path`: comptime metadata
pub fn init(comptime config: Config) type {
    // Build the Fn enum at comptime from config.fn_names
    const TagType = std.math.IntFittingRange(0, config.fn_names.len -| 1);
    const Fn = @Enum(TagType, .exhaustive, config.fn_names, &std.simd.iota(TagType, config.fn_names.len));

    // Import the driver types from zcuda
    const driver = @import("zcuda").driver;

    return struct {
        /// Compile-time kernel name.
        pub const name: []const u8 = config.name;

        /// Path to the compiled PTX file.
        pub const ptx_path: []const u8 = config.ptx_path;

        /// Path to the kernel source file (if provided).
        pub const source_path: ?[]const u8 = config.source_path;

        /// Compile-time enum of all exported kernel function names.
        /// Typos in function names cause build errors, not runtime errors.
        pub const Function = Fn;

        /// Load the kernel's PTX module.
        ///
        /// Auto-detects the loading mode:
        /// - **Embedded mode** (`ptx_data != null`): uses compile-time embedded data, zero file I/O
        /// - **Disk mode** (`ptx_data == null`): reads PTX from `ptx_path` at runtime
        ///
        /// The caller must call `module.deinit()` when done.
        pub fn load(ctx: *const driver.CudaContext, allocator: std.mem.Allocator) !driver.CudaModule {
            if (config.ptx_data) |embedded| {
                // ── Embedded mode: use compile-time data, no file I/O ──
                return ctx.loadModule(embedded);
            }
            // ── Disk mode: read PTX file at runtime ──
            const Io = std.Io;
            var threaded = Io.Threaded.init(allocator, .{ .environ = .empty });
            const io = threaded.io();

            const file = Io.Dir.openFile(.cwd(), io, config.ptx_path, .{}) catch |err| {
                std.log.err("[zcuda bridge] Failed to open PTX '{s}': {}", .{ config.ptx_path, err });
                return err;
            };
            defer file.close(io);

            const ptx_bytes = blk: {
                var read_buf: [8192]u8 = undefined;
                var r = file.reader(io, &read_buf);
                const size = r.getSize() catch |err| {
                    std.log.err("[zcuda bridge] Failed to get PTX size '{s}': {}", .{ config.ptx_path, err });
                    return err;
                };
                break :blk r.interface.readAlloc(allocator, size) catch |err| {
                    std.log.err("[zcuda bridge] Failed to read PTX '{s}': {}", .{ config.ptx_path, err });
                    return err;
                };
            };
            defer allocator.free(ptx_bytes);

            return ctx.loadModule(ptx_bytes);
        }

        /// Load PTX from an in-memory buffer (useful for embedded PTX or NVRTC output).
        pub fn loadFromPtx(ctx: *const driver.CudaContext, ptx_data: []const u8) !driver.CudaModule {
            return ctx.loadModule(ptx_data);
        }

        /// Get a function handle from a loaded module by compile-time enum name.
        ///
        /// Example: `const f = try kernel.getFunction(module, .vectorAdd);`
        ///
        /// If the function name doesn't match any exported function in the kernel,
        /// you get a compile error (not a runtime error).
        pub fn getFunction(module: driver.CudaModule, comptime func: Fn) !driver.CudaFunction {
            const func_name: [:0]const u8 = @tagName(func);
            return module.getFunction(func_name.ptr);
        }

        /// Get a function handle by runtime string name (escape hatch).
        /// Prefer `getFunction` for compile-time safety.
        pub fn getFunctionByName(module: driver.CudaModule, func_name: [*:0]const u8) !driver.CudaFunction {
            return module.getFunction(func_name);
        }
    };
}
