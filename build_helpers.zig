/// build_helpers.zig — Build-time kernel discovery and bridge module generation
///
/// Provides two APIs for build.zig:
///   1. `discoverKernels(b, root_dir)` — recursively scan for .zig files
///      containing `export fn`, identify them as kernels
///   2. `addBridgeModules(b, kernels, opts)` — compile to PTX, generate one-liner
///      bridge modules, wire imports
///
/// This file only depends on `std` — build.zig imports it directly via
/// `@import("build_helpers.zig")`.
///
/// ## Usage in build.zig
///
/// ```zig
/// const bridge = @import("build_helpers.zig");
///
/// const kernel_dir = b.option([]const u8, "kernel-dir",
///     "Root dir for kernel discovery") orelse "src/kernel/";
/// const kernels = bridge.discoverKernels(b, kernel_dir);
/// bridge.addBridgeModules(b, kernels, .{
///     .embed_ptx = embed_ptx,
///     .zcuda_bridge_mod = zcuda_bridge_mod,
///     .zcuda_mod = zcuda_mod,
///     .device_mod = device_mod,
///     .nvptx_target = nvptx_target,
///     .kernel_step = kernel_step,
///     .target = target,
///     .optimize = optimize,
/// });
/// ```
const std = @import("std");
const Io = std.Io;
const Dir = Io.Dir;
const Build = std.Build;
const Allocator = std.mem.Allocator;

// ━━━ Types ━━━

pub const KernelInfo = struct {
    /// Kernel module name (stem of the .zig file, e.g. "kernel_vector_add")
    name: []const u8,
    /// Relative path to the kernel source file
    path: []const u8,
    /// Exported function names extracted from the source
    export_fns: []const []const u8,
};

pub const BridgeOptions = struct {
    /// When true, embed PTX data in the binary via @embedFile (production mode).
    embed_ptx: bool = false,
    /// The zcuda_bridge module (contains bridge_gen.zig)
    zcuda_bridge_mod: *Build.Module,
    /// The zcuda module (driver API)
    zcuda_mod: *Build.Module,
    /// Device intrinsics module for GPU kernel compilation
    device_mod: *Build.Module,
    /// Resolved nvptx64 target for kernel compilation
    nvptx_target: Build.ResolvedTarget,
    /// Parent step that all kernel compilations depend on
    kernel_step: *Build.Step,
    /// Host target
    target: Build.ResolvedTarget,
    /// Host optimization level
    optimize: std.builtin.OptimizeMode,
};

pub const BridgeResult = struct {
    modules: []const BridgeEntry,
};

pub const BridgeEntry = struct {
    name: []const u8,
    module: *Build.Module,
    /// The PTX install step for this bridge (only set for disk-mode LLVM-compiled kernels).
    /// Add this as a dependency of any step that needs the PTX file at runtime.
    install_step: ?*Build.Step = null,
};

// ━━━ Core API 1: Recursive kernel discovery ━━━

/// Recursively scan `root_dir` for all `.zig` files containing `export fn` declarations.
/// Returns a list of discovered kernels with their exported function names.
///
/// Detection is content-based (not filename-based): any `.zig` file with at least one
/// `export fn` is considered a kernel. Files without `export fn` are silently skipped.
pub fn discoverKernels(b: *Build, root_dir: []const u8) []const KernelInfo {
    const allocator = b.allocator;
    const io = b.graph.io;
    var result: std.ArrayList(KernelInfo) = .empty;
    discoverRecursive(allocator, io, &result, root_dir);
    return result.toOwnedSlice(allocator) catch &.{};
}

fn discoverRecursive(
    allocator: Allocator,
    io: Io,
    result: *std.ArrayList(KernelInfo),
    dir_path: []const u8,
) void {
    // Open the directory for iteration
    const dir = Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return;
    var iter = dir.iterate();
    while (iter.next(io) catch null) |entry| {
        const full_path = std.fmt.allocPrint(
            allocator,
            "{s}/{s}",
            .{ dir_path, entry.name },
        ) catch continue;

        switch (entry.kind) {
            .directory => {
                discoverRecursive(allocator, io, result, full_path);
            },
            .file => {
                if (!std.mem.endsWith(u8, entry.name, ".zig")) continue;

                // Content-based detection: scan for export fn declarations
                const fns = scanExportFns(allocator, io, full_path);
                if (fns.len == 0) continue;

                const stem = entry.name[0 .. entry.name.len - 4]; // strip .zig
                result.append(allocator, .{
                    .name = allocator.dupe(u8, stem) catch continue,
                    .path = full_path,
                    .export_fns = fns,
                }) catch continue;
            },
            else => {},
        }
    }
}

/// Scan a source file line-by-line, extracting function names from
/// `export fn name(` and `pub export fn name(` patterns.
fn scanExportFns(allocator: Allocator, io: Io, path: []const u8) []const []const u8 {
    var fns: std.ArrayList([]const u8) = .empty;

    // Read the entire file at once (kernel files are small)
    var buf: [16 * 1024]u8 = undefined;
    const content = Dir.cwd().readFile(io, path, &buf) catch return &.{};

    // Parse line by line
    var line_start: usize = 0;
    while (line_start < content.len) {
        const line_end = std.mem.indexOfScalarPos(u8, content, line_start, '\n') orelse content.len;
        const line = content[line_start..line_end];
        line_start = line_end + 1;

        const trimmed = std.mem.trim(u8, line, " \t\r");
        const keyword_export = "export fn ";
        const keyword_pub_export = "pub export fn ";
        const offset: usize = if (std.mem.startsWith(u8, trimmed, keyword_export))
            keyword_export.len
        else if (std.mem.startsWith(u8, trimmed, keyword_pub_export))
            keyword_pub_export.len
        else
            continue;
        const rest = trimmed[offset..];
        const paren = std.mem.indexOf(u8, rest, "(") orelse continue;
        if (paren == 0) continue;
        fns.append(allocator, allocator.dupe(u8, rest[0..paren]) catch continue) catch continue;
    }
    return fns.toOwnedSlice(allocator) catch &.{};
}

// ━━━ Core API 2: Bridge module generation ━━━

/// Compile all discovered kernels to PTX and generate bridge modules.
///
/// For each kernel:
///   1. Compile .zig → nvptx64 → install to zig-out/bin/kernel/*.ptx
///   2. Generate a one-liner bridge source via addWriteFiles
///   3. Create a module with zcuda_bridge + zcuda imports
///   4. (embed-ptx mode) Inject PTX data via addAnonymousImport
pub fn addBridgeModules(b: *Build, kernels: []const KernelInfo, opts: BridgeOptions) BridgeResult {
    const bridge_wf = b.addWriteFiles();
    var entries: std.ArrayList(BridgeEntry) = .empty;

    for (kernels) |k| {
        // 1. Compile kernel → PTX, or use a pre-built .ptx file if present.
        //    A hand-crafted .ptx file next to the .zig source bypasses LLVM codegen
        //    (workaround for LLVM NVPTX WMMA register naming bug and similar issues).
        const prebuilt_ptx_path = if (std.mem.endsWith(u8, k.path, ".zig"))
            std.fmt.allocPrint(b.allocator, "{s}.ptx", .{k.path[0 .. k.path.len - 4]}) catch null
        else
            null;

        const has_prebuilt = if (prebuilt_ptx_path) |pp|
            prebuiltPtxExists(b, pp)
        else
            false;

        const ptx_output: Build.LazyPath = if (has_prebuilt) blk: {
            // Use the pre-built .ptx file directly — skip LLVM compilation entirely.
            break :blk b.path(prebuilt_ptx_path.?);
        } else blk: {
            // Compile the Zig kernel source to PTX via LLVM NVPTX.
            const kernel_obj = b.addObject(.{
                .name = k.name,
                .root_module = b.createModule(.{
                    .root_source_file = b.path(k.path),
                    .target = opts.nvptx_target,
                    .optimize = .ReleaseFast,
                }),
            });
            kernel_obj.root_module.addImport("zcuda_kernel", opts.device_mod);
            break :blk kernel_obj.getEmittedAsm();
        };

        const install_ptx = b.addInstallFile(
            ptx_output,
            std.fmt.allocPrint(b.allocator, "bin/kernel/{s}.ptx", .{k.name}) catch continue,
        );
        opts.kernel_step.dependOn(&install_ptx.step);

        // Per-kernel build step
        const per_step = b.step(
            std.fmt.allocPrint(b.allocator, "kernel-{s}", .{k.name}) catch continue,
            std.fmt.allocPrint(b.allocator, "Compile kernel: {s}", .{k.name}) catch continue,
        );
        per_step.dependOn(&install_ptx.step);

        // 2. Generate bridge source (explicit re-exports — `usingnamespace` removed in Zig 0.14+)
        //    Pre-built PTX kernels always embed their data at compile time (no disk I/O at runtime).
        //    LLVM-compiled kernels use the embed_ptx option (false = load from zig-out at runtime).
        const fn_list = formatFnNames(b.allocator, k.export_fns);
        const should_embed = has_prebuilt or opts.embed_ptx;
        const bridge_src = if (should_embed)
            std.fmt.allocPrint(b.allocator,
                \\const _b = @import("zcuda_bridge").init(.{{
                \\    .name = "{s}",
                \\    .ptx_path = "zig-out/bin/kernel/{s}.ptx",
                \\    .fn_names = {s},
                \\    .ptx_data = @embedFile("ptx_data"),
                \\}});
                \\pub const load = _b.load;
                \\pub const loadFromPtx = _b.loadFromPtx;
                \\pub const getFunction = _b.getFunction;
                \\pub const getFunctionByName = _b.getFunctionByName;
                \\pub const Function = _b.Function;
                \\pub const name = _b.name;
                \\pub const ptx_path = _b.ptx_path;
                \\pub const source_path = _b.source_path;
            , .{ k.name, k.name, fn_list }) catch continue
        else
            std.fmt.allocPrint(b.allocator,
                \\const _b = @import("zcuda_bridge").init(.{{
                \\    .name = "{s}",
                \\    .ptx_path = "zig-out/bin/kernel/{s}.ptx",
                \\    .fn_names = {s},
                \\}});
                \\pub const load = _b.load;
                \\pub const loadFromPtx = _b.loadFromPtx;
                \\pub const getFunction = _b.getFunction;
                \\pub const getFunctionByName = _b.getFunctionByName;
                \\pub const Function = _b.Function;
                \\pub const name = _b.name;
                \\pub const ptx_path = _b.ptx_path;
                \\pub const source_path = _b.source_path;
            , .{ k.name, k.name, fn_list }) catch continue;

        const gen_path = std.fmt.allocPrint(b.allocator, "bridges/{s}.zig", .{k.name}) catch continue;
        const gen_file = bridge_wf.add(gen_path, bridge_src);

        // 3. Create module + wiring
        const mod = b.createModule(.{
            .root_source_file = gen_file,
            .target = opts.target,
            .optimize = opts.optimize,
        });
        mod.addImport("zcuda_bridge", opts.zcuda_bridge_mod);
        mod.addImport("zcuda", opts.zcuda_mod);

        // 4. Embed PTX data: always for pre-built PTX, optionally for LLVM-compiled PTX
        if (should_embed) {
            mod.addAnonymousImport("ptx_data", .{
                .root_source_file = ptx_output,
            });
        }

        entries.append(b.allocator, .{ .name = k.name, .module = mod, .install_step = if (!should_embed) &install_ptx.step else null }) catch continue;
    }

    return .{
        .modules = entries.toOwnedSlice(b.allocator) catch &.{},
    };
}

/// Find a bridge module by name from the result list.
pub fn findBridge(modules: []const BridgeEntry, name: []const u8) ?*Build.Module {
    for (modules) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry.module;
    }
    return null;
}

/// Find the PTX install step for a disk-mode bridge (returns null for embedded-mode bridges).
pub fn findBridgeInstallStep(modules: []const BridgeEntry, name: []const u8) ?*Build.Step {
    for (modules) |entry| {
        if (std.mem.eql(u8, entry.name, name)) return entry.install_step;
    }
    return null;
}

// ━━━ Internal helpers ━━━

/// Check whether a pre-built PTX file exists on disk (build-time check).
/// Used to detect hand-crafted .ptx files that should bypass LLVM compilation.
fn prebuiltPtxExists(b: *Build, ptx_path: []const u8) bool {
    const io = b.graph.io;
    var buf: [16]u8 = undefined;
    // Try to read the first few bytes; success means the file exists.
    const content = Dir.cwd().readFile(io, ptx_path, &buf) catch return false;
    _ = content;
    return true;
}

/// Format an array of function names as a Zig array literal string: `&.{"fn1", "fn2"}`
fn formatFnNames(allocator: Allocator, fn_names: []const []const u8) []const u8 {
    if (fn_names.len == 0) return "&.{}";
    if (fn_names.len == 1) {
        return std.fmt.allocPrint(allocator, "&.{{\"{s}\"}}", .{fn_names[0]}) catch "&.{}";
    }
    // Build multi-fn list via successive allocPrint calls (simple, build-time only)
    var result = std.fmt.allocPrint(allocator, "&.{{\"{s}\"", .{fn_names[0]}) catch return "&.{}";
    for (fn_names[1..]) |fname| {
        result = std.fmt.allocPrint(allocator, "{s}, \"{s}\"", .{ result, fname }) catch return "&.{}";
    }
    return std.fmt.allocPrint(allocator, "{s}}}", .{result}) catch "&.{}";
}

// ━━━ GPU Architecture Helpers (public — usable by downstream build.zig) ━━━

/// Parse a GPU architecture string such as `"sm_80"` into the SM version integer (e.g. 80).
///
/// Accepts the standard `"sm_XX"` / `"sm_XXX"` format produced by `-Dgpu-arch=`.
/// Panics with a descriptive message on invalid input.
///
/// ## Example
///
/// ```zig
/// const bridge = @import("zcuda").build_helpers;
/// const sm = bridge.parseSmVersion(gpu_arch_option); // e.g. 86
/// ```
pub fn parseSmVersion(arch: []const u8) u32 {
    if (arch.len >= 4 and std.mem.eql(u8, arch[0..3], "sm_")) {
        return std.fmt.parseInt(u32, arch[3..], 10) catch {
            @panic("Invalid gpu-arch: expected digits after 'sm_' (e.g. sm_80, sm_86)");
        };
    }
    @panic("Invalid gpu-arch format. Expected 'sm_XX' (e.g. sm_70, sm_80, sm_90)");
}

/// Map an SM version integer to the corresponding nvptx64 CPU model.
///
/// The CPU model controls both the `.target sm_XX` directive and the PTX ISA version
/// written into the compiled PTX output.
///
/// Supported: sm_52, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100.
///
/// ## Example
///
/// ```zig
/// const bridge = @import("zcuda").build_helpers;
/// const model  = bridge.smVersionToCpuModel(86); // &std.Target.nvptx.cpu.sm_86
/// ```
pub fn smVersionToCpuModel(sm: u32) *const std.Target.Cpu.Model {
    const cpu = std.Target.nvptx.cpu;
    return switch (sm) {
        52  => &cpu.sm_52,
        60  => &cpu.sm_60,
        70  => &cpu.sm_70,
        75  => &cpu.sm_75,
        80  => &cpu.sm_80,
        86  => &cpu.sm_86,
        89  => &cpu.sm_89,
        90  => &cpu.sm_90,
        100 => &cpu.sm_100,
        else => @panic("Unsupported SM version. Supported: sm_52, sm_60, sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100"),
    };
}

/// Resolve an nvptx64 `Build.ResolvedTarget` from a `"-Dgpu-arch=sm_XX"` string.
///
/// This is the **recommended one-call API** for downstream build scripts — it replaces
/// the three-step boilerplate of parse → cpu model → resolveTargetQuery.
///
/// ## Example (downstream build.zig)
///
/// ```zig
/// const bridge   = @import("zcuda").build_helpers;
/// const gpu_arch = b.option([]const u8, "gpu-arch", "Target SM arch") orelse "sm_80";
///
/// // One call instead of 8 lines of boilerplate:
/// const nvptx_target = bridge.resolveNvptxTarget(b, gpu_arch);
/// ```
pub fn resolveNvptxTarget(b: *Build, gpu_arch: []const u8) Build.ResolvedTarget {
    const sm = parseSmVersion(gpu_arch);
    return b.resolveTargetQuery(.{
        .cpu_arch  = .nvptx64,
        .os_tag    = .cuda,
        .abi       = .none,
        .cpu_model = .{ .explicit = smVersionToCpuModel(sm) },
    });
}

/// Create the device intrinsics module (`zcuda_kernel`) for the given nvptx64 target.
///
/// This is the **recommended API** for downstream build scripts — it hides the
/// internal `src/kernel/device.zig` path and wires `sm_version` build_options
/// automatically.
///
/// The returned module should be passed as `.device_mod` in `BridgeOptions`.
/// It is compiled for `nvptx_target` (GPU side), not the host.
///
/// ## Example (downstream build.zig)
///
/// ```zig
/// const bridge       = @import("zcuda").build_helpers;
/// const gpu_arch     = b.option([]const u8, "gpu-arch", "Target SM arch") orelse "sm_80";
/// const nvptx_target = bridge.resolveNvptxTarget(b, gpu_arch);
/// const device_mod   = bridge.makeDeviceModule(b, zcuda_dep, nvptx_target, gpu_arch);
///
/// const result = bridge.addBridgeModules(b, kernels, .{
///     .device_mod   = device_mod,
///     .nvptx_target = nvptx_target,
///     // ...
/// });
/// ```
pub fn makeDeviceModule(
    b: *Build,
    zcuda_dep: *Build.Dependency,
    nvptx_target: Build.ResolvedTarget,
    gpu_arch: []const u8,
) *Build.Module {
    const sm = parseSmVersion(gpu_arch);
    const opts = b.addOptions();
    opts.addOption(u32, "sm_version", sm);

    const mod = b.createModule(.{
        .root_source_file = zcuda_dep.path("src/kernel/device.zig"),
        .target   = nvptx_target,
        .optimize = .ReleaseFast,
    });
    mod.addOptions("build_options", opts);
    return mod;
}
