const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const Step = std.Build.Step;

/// Re-export build_helpers for downstream users.
/// Usage in downstream build.zig:
///   const zcuda = @import("zcuda");
///   const kernels = zcuda.build_helpers.discoverKernels(b, "src/kernel/");
pub const build_helpers = @import("build_helpers.zig");

/// Check if a file exists at the given absolute path using faccessat.
fn fileExists(path: []const u8) bool {
    // Use openat with AT_FDCWD to check file existence
    const fd = posix.openat(posix.AT.FDCWD, path, .{}, 0) catch return false;
    posix.close(fd);
    return true;
}

/// Find the CUDA installation path by checking common locations.
fn findCudaPath(allocator: std.mem.Allocator, cuda_path_opt: ?[]const u8) ![]const u8 {
    if (cuda_path_opt) |path| {
        const cuda_h = try std.fmt.allocPrint(allocator, "{s}/include/cuda.h", .{path});
        defer allocator.free(cuda_h);
        if (fileExists(cuda_h)) return path;
        return error.CudaInstallationNotFound;
    }

    // Common CUDA installation paths
    const probable_paths = [_][]const u8{
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-13.1",
        "/opt/cuda",
        "/usr/lib/cuda",
        "/usr",
    };

    for (probable_paths) |path| {
        const cuda_h = try std.fmt.allocPrint(allocator, "{s}/include/cuda.h", .{path});
        defer allocator.free(cuda_h);
        if (fileExists(cuda_h)) return path;
    }

    return error.CudaInstallationNotFound;
}

/// Find the cuDNN include path by checking common locations.
fn findCudnnIncludePath(allocator: std.mem.Allocator) ?[]const u8 {
    const probable_paths = [_][]const u8{
        "/usr/include/x86_64-linux-gnu",
        "/usr/include",
        "/usr/local/cuda/include",
    };

    for (probable_paths) |path| {
        const cudnn_h = std.fmt.allocPrint(allocator, "{s}/cudnn.h", .{path}) catch continue;
        defer allocator.free(cudnn_h);
        if (fileExists(cudnn_h)) return path;
    }
    return null;
}

/// Helper to configure include and library paths for a compilation step module.
fn configurePaths(
    mod: *std.Build.Module,
    cuda_path: []const u8,
    cuda_include: []const u8,
    cudnn_include: ?[]const u8,
    lib_paths: []const []const u8,
    allocator: std.mem.Allocator,
) !void {
    mod.addIncludePath(.{ .cwd_relative = cuda_include });
    if (cudnn_include) |p| {
        mod.addIncludePath(.{ .cwd_relative = p });
    }
    for (lib_paths) |lib_path| {
        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ cuda_path, lib_path });
        if (!fileExists(full_path)) continue;
        mod.addLibraryPath(.{ .cwd_relative = full_path });
    }
    // Also add cuDNN library path
    if (fileExists("/usr/lib/x86_64-linux-gnu")) {
        mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
    }
}

/// Named struct type for library feature flags.
const LibOpts = struct {
    cublas: bool,
    cublaslt: bool,
    curand: bool,
    nvrtc: bool,
    cudnn: bool,
    cusolver: bool,
    cusparse: bool,
    cufft: bool,
    cupti: bool,
    cufile: bool,
    nvtx: bool,
};

/// Link all enabled CUDA libraries to a module.
fn linkLibrariesToModule(mod: *std.Build.Module, opts: LibOpts) void {
    const no_opts = std.Build.Module.LinkSystemLibraryOptions{};
    mod.link_libc = true;
    mod.linkSystemLibrary("cuda", no_opts);
    mod.linkSystemLibrary("cudart", no_opts);
    if (opts.nvrtc) mod.linkSystemLibrary("nvrtc", no_opts);
    if (opts.cublas) mod.linkSystemLibrary("cublas", no_opts);
    if (opts.cublaslt) mod.linkSystemLibrary("cublasLt", no_opts);
    if (opts.curand) mod.linkSystemLibrary("curand", no_opts);
    if (opts.cudnn) mod.linkSystemLibrary("cudnn", no_opts);
    if (opts.cusolver) mod.linkSystemLibrary("cusolver", no_opts);
    if (opts.cusparse) mod.linkSystemLibrary("cusparse", no_opts);
    if (opts.cufft) mod.linkSystemLibrary("cufft", no_opts);
    if (opts.cupti) mod.linkSystemLibrary("cupti", no_opts);
    if (opts.cufile) mod.linkSystemLibrary("cufile", no_opts);
    if (opts.nvtx) mod.linkSystemLibrary("nvToolsExt", no_opts);
}

/// Link all enabled CUDA libraries to a compile step.
fn linkLibraries(step: *std.Build.Step.Compile, opts: LibOpts) void {
    linkLibrariesToModule(step.root_module, opts);
}

pub fn build(b: *std.Build) !void {
    if (builtin.os.tag == .windows) {
        @panic("Windows support is not yet available");
    }

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- Build options ---
    const cuda_path_opt = b.option([]const u8, "cuda-path", "Path to CUDA installation (default: auto-detect)");
    const cuda_path: ?[]const u8 = findCudaPath(b.allocator, cuda_path_opt) catch null;
    const cuda_include: ?[]const u8 = if (cuda_path) |cp|
        try std.fmt.allocPrint(b.allocator, "{s}/include", .{cp})
    else
        null;
    const cudnn_include = findCudnnIncludePath(b.allocator);

    // Feature flags (default: core modules enabled)
    const enable_cublas = b.option(bool, "cublas", "Enable cuBLAS bindings (default: true)") orelse true;
    const enable_cublaslt = b.option(bool, "cublaslt", "Enable cuBLAS LT bindings (default: true)") orelse true;
    const enable_curand = b.option(bool, "curand", "Enable cuRAND bindings (default: true)") orelse true;
    const enable_nvrtc = b.option(bool, "nvrtc", "Enable NVRTC bindings (default: true)") orelse true;
    const enable_cudnn = b.option(bool, "cudnn", "Enable cuDNN bindings (default: false)") orelse false;
    const enable_cusolver = b.option(bool, "cusolver", "Enable cuSOLVER bindings (default: false)") orelse false;
    const enable_cusparse = b.option(bool, "cusparse", "Enable cuSPARSE bindings (default: false)") orelse false;
    const enable_cufft = b.option(bool, "cufft", "Enable cuFFT bindings (default: false)") orelse false;
    const enable_cupti = b.option(bool, "cupti", "Enable CUPTI bindings (default: false)") orelse false;
    const enable_cufile = b.option(bool, "cufile", "Enable cuFile (GPUDirect Storage) bindings (default: false)") orelse false;
    const enable_nvtx = b.option(bool, "nvtx", "Enable NVTX bindings (default: false)") orelse false;

    // ── Union-of-libs scan ──
    // Scan all example/integration-example .libs strings and OR into the global
    // enable_* flags so that zcuda_mod's build_options satisfy every @compileError
    // guard in src/cuda.zig WITHOUT needing per-example zcuda modules.
    // Keeping a single zcuda_mod avoids the "file exists in modules zcuda/zcuda0" error
    // that occurs when bridge modules are shared across examples.
    const all_example_libs: []const u8 =
        "cublas,cublaslt,curand,nvrtc" // examples/cublas/, examples/curand/, examples/nvrtc/
        ++ ",cufft"                     // examples/cufft/, integration E_cuFFT_Pipelines
        ++ ",cusolver,cusparse"         // examples/cusolver/, examples/cusparse/
        ++ ",cudnn,nvtx,cupti,cufile";  // optional extras
    // Override: if any example/integration needs a lib, also enable it in zcuda_mod.
    // Using std.mem.indexOf on the aggregated string is evaluated at build time.
    const eff_cufft    = enable_cufft    or (std.mem.indexOf(u8, all_example_libs, "cufft")    != null);
    const eff_cusolver = enable_cusolver or (std.mem.indexOf(u8, all_example_libs, "cusolver") != null);
    const eff_cusparse = enable_cusparse or (std.mem.indexOf(u8, all_example_libs, "cusparse") != null);
    const eff_cudnn    = enable_cudnn    or (std.mem.indexOf(u8, all_example_libs, "cudnn")    != null);
    const eff_cupti    = enable_cupti    or (std.mem.indexOf(u8, all_example_libs, "cupti")    != null);
    const eff_cufile   = enable_cufile   or (std.mem.indexOf(u8, all_example_libs, "cufile")   != null);
    const eff_nvtx     = enable_nvtx     or (std.mem.indexOf(u8, all_example_libs, "nvtx")     != null);

    // Library paths to search
    const lib_paths = [_][]const u8{
        "lib",
        "lib64",
        "lib/x64",
        "lib64/stubs",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    };

    // Build options passed to source code
    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_cublas", enable_cublas);
    build_options.addOption(bool, "enable_cublaslt", enable_cublaslt);
    build_options.addOption(bool, "enable_curand", enable_curand);
    build_options.addOption(bool, "enable_nvrtc", enable_nvrtc);
    build_options.addOption(bool, "enable_cudnn",    eff_cudnn);
    build_options.addOption(bool, "enable_cusolver", eff_cusolver);
    build_options.addOption(bool, "enable_cusparse", eff_cusparse);
    build_options.addOption(bool, "enable_cufft",    eff_cufft);
    build_options.addOption(bool, "enable_cupti",    eff_cupti);
    build_options.addOption(bool, "enable_cufile",   eff_cufile);
    build_options.addOption(bool, "enable_nvtx",     eff_nvtx);

    const lib_opts = LibOpts{
        .cublas = enable_cublas,
        .cublaslt = enable_cublaslt,
        .curand = enable_curand,
        .nvrtc = enable_nvrtc,
        .cudnn = enable_cudnn,
        .cusolver = enable_cusolver,
        .cusparse = enable_cusparse,
        .cufft = enable_cufft,
        .cupti = enable_cupti,
        .cufile = enable_cufile,
        .nvtx = enable_nvtx,
    };

    // ── Export zcuda as a Zig package for downstream consumers ──
    // Host-side features require CUDA installation.
    // Kernel compilation (compile-kernels) works without CUDA.
    // NOTE: System libraries (cuda, cudart, etc.) are NOT linked here —
    // linking must happen on compile steps with known targets, not on
    // exported modules. Downstream consumers call linkLibraries() on
    // their own compile steps. Internal tests/examples do the same below.
    const zcuda_mod = b.addModule("zcuda", .{
        .root_source_file = b.path("src/cuda.zig"),
    });
    zcuda_mod.addOptions("build_options", build_options);
    if (cuda_path) |cp| {
        try configurePaths(zcuda_mod, cp, cuda_include.?, cudnn_include, &lib_paths, b.allocator);
    }


    // ── Export zcuda_bridge module for Way 5 kernel bridges ──
    // Downstream users import this to create type-safe kernel bridge modules.
    // See src/kernel/bridge_gen.zig for usage documentation.
    const zcuda_bridge_mod = b.addModule("zcuda_bridge", .{
        .root_source_file = b.path("src/kernel/bridge_gen.zig"),
    });
    zcuda_bridge_mod.addImport("zcuda", zcuda_mod);

    // ── Export zcuda as a Zig package for downstream consumers ──

    // --- Unit tests (test/unit/*) ---
    const unit_test_step = b.step("test-unit", "Run unit tests");

    // Core tests (always available — only need cuda driver/runtime)
    const core_unit_tests = [_][]const u8{
        "test/unit/types_test.zig",
        "test/unit/driver_test.zig",
        "test/unit/runtime_test.zig",
        "test/unit/nvrtc_test.zig",
    };

    for (core_unit_tests) |test_path| {
        const unit = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(test_path),
                .target = target,
                .optimize = optimize,
            }),
        });
        unit.root_module.addImport("zcuda", zcuda_mod);
        unit.root_module.addOptions("build_options", build_options);
        if (cuda_path) |cp| {
            try configurePaths(unit.root_module, cp, cuda_include.?, cudnn_include, &lib_paths, b.allocator);
        }
        linkLibraries(unit, lib_opts);

        // GPU-linked tests use addGpuTestRun to avoid Zig's --listen=- IPC
        // protocol, which conflicts with CUDA's dynamic library.
        // Pure Zig tests (types_test) use the normal addRunArtifact.
        const is_gpu_test = !std.mem.endsWith(u8, test_path, "types_test.zig");
        const run_unit = if (is_gpu_test)
            addGpuTestRun(b, unit)
        else
            b.addRunArtifact(unit);
        unit_test_step.dependOn(&run_unit.step);
    }

    // Kernel module unit tests (no CUDA dependency — pure Zig logic)
    {
        // Build options for kernel modules that depend on SM version
        // (Default sm_80 — tests only check declarations/comptime, not GPU runtime)
        const kernel_test_options = b.addOptions();
        kernel_test_options.addOption(u32, "sm_version", 80);

        // Helper: create a module for a kernel source file with build_options
        const KernelTestEntry = struct {
            test_path: []const u8,
            import_name: []const u8,
            source_path: []const u8,
            needs_build_options: bool,
            needs_zcuda: bool = false,
        };

        const kernel_tests = [_]KernelTestEntry{
            // Pure-Zig type tests (no GPU needed)
            .{ .test_path = "test/unit/kernel/kernel_shared_types_test.zig", .import_name = "shared_types", .source_path = "src/kernel/shared_types.zig", .needs_build_options = false },
            .{ .test_path = "test/unit/kernel/kernel_types_test.zig", .import_name = "shared_types", .source_path = "src/kernel/shared_types.zig", .needs_build_options = false },
            .{ .test_path = "test/unit/kernel/kernel_arch_test.zig", .import_name = "arch", .source_path = "src/kernel/arch.zig", .needs_build_options = false },
            // Correctness tests (need build_options for SM version)
            .{ .test_path = "test/unit/kernel/kernel_device_types_test.zig", .import_name = "types", .source_path = "src/kernel/types.zig", .needs_build_options = true },
            .{ .test_path = "test/unit/kernel/kernel_device_test.zig", .import_name = "types", .source_path = "src/kernel/types.zig", .needs_build_options = true, .needs_zcuda = true },
            .{ .test_path = "test/unit/kernel/kernel_debug_test.zig", .import_name = "debug", .source_path = "src/kernel/debug.zig", .needs_build_options = true },
            .{ .test_path = "test/unit/kernel/kernel_shared_mem_test.zig", .import_name = "shared_mem", .source_path = "src/kernel/shared_mem.zig", .needs_build_options = true },
            .{ .test_path = "test/unit/kernel/kernel_intrinsics_host_test.zig", .import_name = "intrinsics", .source_path = "src/kernel/intrinsics.zig", .needs_build_options = true },
            .{ .test_path = "test/unit/kernel/kernel_tensor_core_host_test.zig", .import_name = "tensor_core", .source_path = "src/kernel/tensor_core.zig", .needs_build_options = true },
            .{ .test_path = "test/unit/kernel/kernel_grid_stride_test.zig", .import_name = "types", .source_path = "src/kernel/types.zig", .needs_build_options = true },
        };

        for (kernel_tests) |entry| {
            const src_mod = b.createModule(.{
                .root_source_file = b.path(entry.source_path),
                .target = target,
                .optimize = optimize,
            });
            if (entry.needs_build_options) {
                src_mod.addOptions("build_options", kernel_test_options);
            }

            const t = b.addTest(.{
                .root_module = b.createModule(.{
                    .root_source_file = b.path(entry.test_path),
                    .target = target,
                    .optimize = optimize,
                }),
            });
            t.root_module.addImport(entry.import_name, src_mod);
            if (entry.needs_zcuda) {
                t.root_module.addImport("zcuda", zcuda_mod);
                t.root_module.addImport("test_helpers", b.createModule(.{
                    .root_source_file = b.path("test/helpers.zig"),
                    .imports = &.{.{ .name = "zcuda", .module = zcuda_mod }},
                }));
                linkLibrariesToModule(t.root_module, lib_opts);
            }

            const run = b.addRunArtifact(t);
            unit_test_step.dependOn(&run.step);
        }
    }

    // Conditional unit tests (require specific library flags)
    const ConditionalTest = struct { path: []const u8, enabled: bool };
    const conditional_unit_tests = [_]ConditionalTest{
        .{ .path = "test/unit/nvtx_test.zig", .enabled = enable_nvtx },
        .{ .path = "test/unit/cublas_test.zig", .enabled = enable_cublas },
        .{ .path = "test/unit/cublaslt_test.zig", .enabled = enable_cublaslt },
        .{ .path = "test/unit/curand_test.zig", .enabled = enable_curand },
        .{ .path = "test/unit/cudnn_test.zig", .enabled = enable_cudnn },
        .{ .path = "test/unit/cusolver_test.zig", .enabled = enable_cusolver },
        .{ .path = "test/unit/cusparse_test.zig", .enabled = enable_cusparse },
        .{ .path = "test/unit/cufft_test.zig", .enabled = enable_cufft },
    };

    for (conditional_unit_tests) |ct| {
        if (!ct.enabled) continue;
        const unit = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(ct.path),
                .target = target,
                .optimize = optimize,
            }),
        });
        unit.root_module.addImport("zcuda", zcuda_mod);
        unit.root_module.addOptions("build_options", build_options);
        if (cuda_path) |cp| {
            try configurePaths(unit.root_module, cp, cuda_include.?, cudnn_include, &lib_paths, b.allocator);
        }
        linkLibraries(unit, lib_opts);

        const run_unit = addGpuTestRun(b, unit);
        unit_test_step.dependOn(&run_unit.step);
    }

    // --- Integration tests (test/integration/*) ---
    const integration_test_step = b.step("test-integration", "Run integration tests");

    const ConditionalIntTest = struct { path: []const u8, enabled: bool };
    const integration_tests = [_]ConditionalIntTest{
        .{ .path = "test/integration/gemm_roundtrip_test.zig", .enabled = enable_cublas },
        .{ .path = "test/integration/conv_pipeline_test.zig", .enabled = enable_cudnn },
        .{ .path = "test/integration/lu_solve_test.zig", .enabled = enable_cusolver },
        .{ .path = "test/integration/jit_kernel_test.zig", .enabled = enable_nvrtc },
        .{ .path = "test/integration/fft_roundtrip_test.zig", .enabled = enable_cufft },
        .{ .path = "test/integration/curand_fft_test.zig", .enabled = enable_curand and enable_cufft },
        .{ .path = "test/integration/syrk_geam_test.zig", .enabled = enable_cublas },
        .{ .path = "test/integration/svd_reconstruct_test.zig", .enabled = enable_cusolver },
        .{ .path = "test/integration/sparse_pipeline_test.zig", .enabled = enable_cusparse },
        .{ .path = "test/integration/conv_relu_test.zig", .enabled = enable_cudnn },
        .{ .path = "test/integration/kernel/kernel_pipeline_test.zig", .enabled = true },
        // Kernel module integration tests (GPU correctness verification)
        .{ .path = "test/integration/kernel/kernel_intrinsics_gpu_test.zig", .enabled = true },
        .{ .path = "test/integration/kernel/kernel_reduction_test.zig", .enabled = true },
        .{ .path = "test/integration/kernel/kernel_memory_lifecycle_test.zig", .enabled = true },
        .{ .path = "test/integration/kernel/kernel_event_timing_test.zig", .enabled = true },
        .{ .path = "test/integration/kernel/kernel_shared_mem_gpu_test.zig", .enabled = true },
        .{ .path = "test/integration/kernel/kernel_softmax_test.zig", .enabled = true },
    };

    for (integration_tests) |ct| {
        if (!ct.enabled) continue;
        const integration = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(ct.path),
                .target = target,
                .optimize = optimize,
            }),
        });
        integration.root_module.addImport("zcuda", zcuda_mod);
        integration.root_module.addImport("test_helpers", b.createModule(.{
            .root_source_file = b.path("test/helpers.zig"),
            .imports = &.{.{ .name = "zcuda", .module = zcuda_mod }},
        }));
        integration.root_module.addOptions("build_options", build_options);
        if (cuda_path) |cp| {
            try configurePaths(integration.root_module, cp, cuda_include.?, cudnn_include, &lib_paths, b.allocator);
        }
        linkLibraries(integration, lib_opts);

        const run_integration = addGpuTestRun(b, integration);
        integration_test_step.dependOn(&run_integration.step);
    }

    // --- Combined test step ---
    const test_step = b.step("test", "Run all tests (unit + integration)");
    test_step.dependOn(unit_test_step);
    test_step.dependOn(integration_test_step);

    // --- Examples ---
    const examples = [_]struct { name: []const u8, desc: []const u8, libs: []const u8 }{
        // basics/
        .{ .name = "basics/vector_add", .desc = "Vector addition via JIT kernel", .libs = "" },
        .{ .name = "basics/device_info", .desc = "GPU device info and context config", .libs = "" },
        .{ .name = "basics/event_timing", .desc = "Event-based GPU timing & bandwidth", .libs = "" },
        .{ .name = "basics/streams", .desc = "Multi-stream concurrent execution", .libs = "" },
        .{ .name = "basics/peer_to_peer", .desc = "Multi-GPU peer access demo", .libs = "" },
        .{ .name = "basics/constant_memory", .desc = "GPU constant memory polynomial eval", .libs = "" },
        .{ .name = "basics/struct_kernel", .desc = "Pass Zig extern struct to GPU kernel", .libs = "" },
        .{ .name = "basics/kernel_attributes", .desc = "Query kernel registers/shared/occupancy", .libs = "" },
        // cublas/
        .{ .name = "cublas/axpy", .desc = "cuBLAS SAXPY: y = alpha*x + y", .libs = "cublas" },
        .{ .name = "cublas/dot", .desc = "cuBLAS dot product", .libs = "cublas" },
        .{ .name = "cublas/nrm2_asum", .desc = "cuBLAS L1/L2 vector norms", .libs = "cublas" },
        .{ .name = "cublas/scal", .desc = "cuBLAS vector scaling", .libs = "cublas" },
        .{ .name = "cublas/amax_amin", .desc = "cuBLAS max/min absolute value index", .libs = "cublas" },
        .{ .name = "cublas/swap_copy", .desc = "cuBLAS vector swap and copy", .libs = "cublas" },
        .{ .name = "cublas/cosine_similarity", .desc = "Cosine similarity via cuBLAS L1 ops", .libs = "cublas" },
        .{ .name = "cublas/gemv", .desc = "cuBLAS matrix-vector multiply (SGEMV)", .libs = "cublas" },
        .{ .name = "cublas/gemm", .desc = "cuBLAS matrix-matrix multiply (SGEMM)", .libs = "cublas" },
        .{ .name = "cublas/gemm_batched", .desc = "cuBLAS strided batched GEMM", .libs = "cublas" },
        .{ .name = "cublas/symm", .desc = "cuBLAS symmetric matrix multiply", .libs = "cublas" },
        .{ .name = "cublas/trsm", .desc = "cuBLAS triangular solve", .libs = "cublas" },
        .{ .name = "cublas/gemm_ex", .desc = "cuBLAS mixed-precision GemmEx", .libs = "cublas" },
        // curand/
        .{ .name = "curand/distributions", .desc = "cuRAND uniform/normal/Poisson distributions", .libs = "curand" },
        .{ .name = "curand/monte_carlo_pi", .desc = "Monte Carlo Pi estimation with cuRAND", .libs = "curand" },
        .{ .name = "cublas/rot", .desc = "cuBLAS Givens rotation", .libs = "cublas" },
        .{ .name = "cublas/symv_syr", .desc = "cuBLAS symmetric matrix-vector ops", .libs = "cublas" },
        .{ .name = "cublas/trmv_trsv", .desc = "cuBLAS triangular multiply and solve", .libs = "cublas" },
        .{ .name = "cublas/syrk", .desc = "cuBLAS symmetric rank-k update", .libs = "cublas" },
        .{ .name = "cublas/geam", .desc = "cuBLAS matrix add/transpose", .libs = "cublas" },
        .{ .name = "cublas/dgmm", .desc = "cuBLAS diagonal matrix multiply", .libs = "cublas" },
        // cufft/
        .{ .name = "cufft/fft_1d_c2c", .desc = "cuFFT 1D complex-to-complex", .libs = "cufft" },
        .{ .name = "cufft/fft_1d_r2c", .desc = "cuFFT 1D real-to-complex w/ filtering", .libs = "cufft" },
        .{ .name = "cufft/fft_2d", .desc = "cuFFT 2D complex FFT", .libs = "cufft" },
        // curand/
        .{ .name = "curand/generators", .desc = "cuRAND generator comparison", .libs = "curand" },
        // nvrtc/
        .{ .name = "nvrtc/jit_compile", .desc = "NVRTC runtime kernel compilation", .libs = "" },
        .{ .name = "nvrtc/template_kernel", .desc = "NVRTC multi-kernel pipeline", .libs = "" },
        // nvtx/
        .{ .name = "nvtx/profiling", .desc = "NVTX profiling annotations", .libs = "nvtx" },
        // cusolver/
        .{ .name = "cusolver/getrf", .desc = "cuSOLVER LU factorization + solve", .libs = "cusolver" },
        .{ .name = "cusolver/gesvd", .desc = "cuSOLVER singular value decomposition", .libs = "cusolver" },
        .{ .name = "cusolver/potrf", .desc = "cuSOLVER Cholesky factorization + solve", .libs = "cusolver" },
        .{ .name = "cusolver/syevd", .desc = "cuSOLVER eigenvalue decomposition", .libs = "cusolver" },
        .{ .name = "cusolver/geqrf", .desc = "cuSOLVER QR factorization", .libs = "cusolver" },
        // cusparse/
        .{ .name = "cusparse/spmv_csr", .desc = "cuSPARSE SpMV with CSR format", .libs = "cusparse" },
        .{ .name = "cusparse/spmm_csr", .desc = "cuSPARSE SpMM (sparse x dense)", .libs = "cusparse" },
        .{ .name = "cusparse/spmv_coo", .desc = "cuSPARSE SpMV with COO format", .libs = "cusparse" },
        // cublaslt/
        .{ .name = "cublaslt/lt_sgemm", .desc = "cuBLAS LT SGEMM with heuristics", .libs = "cublaslt" },
        // cudnn/
        .{ .name = "cudnn/activation", .desc = "cuDNN activation functions", .libs = "cudnn" },
        .{ .name = "cudnn/pooling_softmax", .desc = "cuDNN pooling + softmax", .libs = "cudnn" },
        .{ .name = "cudnn/conv2d", .desc = "cuDNN conv2d forward", .libs = "cudnn" },
        // cufft/
        .{ .name = "cufft/fft_3d", .desc = "cuFFT 3D complex FFT", .libs = "cufft" },
        // cusparse/
        .{ .name = "cusparse/spgemm", .desc = "cuSPARSE SpGEMM (sparse x sparse)", .libs = "cusparse" },
    };

    for (examples) |ex| {
        const example_path = try std.fmt.allocPrint(b.allocator, "examples/{s}.zig", .{ex.name});

        // Create a dash-separated step name from the path
        var step_name_buf: [128]u8 = undefined;
        const step_name = blk: {
            var i: usize = 0;
            for (ex.name) |ch| {
                if (ch == '/') {
                    step_name_buf[i] = '-';
                } else {
                    step_name_buf[i] = ch;
                }
                i += 1;
            }
            break :blk step_name_buf[0..i];
        };

        const example = b.addExecutable(.{
            .name = step_name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(example_path),
                .target = target,
                .optimize = optimize,
            }),
        });
        example.root_module.addImport("zcuda", zcuda_mod);
        example.root_module.addOptions("build_options", build_options);
        if (cuda_path) |cp| {
            try configurePaths(example.root_module, cp, cuda_include.?, cudnn_include, &lib_paths, b.allocator);
        }
        // Compute per-example lib_opts: enable libraries declared in `libs` field
        var ex_lib_opts = lib_opts;
        if (std.mem.indexOf(u8, ex.libs, "cublas") != null) ex_lib_opts.cublas = true;
        if (std.mem.indexOf(u8, ex.libs, "cublaslt") != null) ex_lib_opts.cublaslt = true;
        if (std.mem.indexOf(u8, ex.libs, "curand") != null) ex_lib_opts.curand = true;
        if (std.mem.indexOf(u8, ex.libs, "nvrtc") != null) ex_lib_opts.nvrtc = true;
        if (std.mem.indexOf(u8, ex.libs, "cudnn") != null) ex_lib_opts.cudnn = true;
        if (std.mem.indexOf(u8, ex.libs, "cusolver") != null) ex_lib_opts.cusolver = true;
        if (std.mem.indexOf(u8, ex.libs, "cusparse") != null) ex_lib_opts.cusparse = true;
        if (std.mem.indexOf(u8, ex.libs, "cufft") != null) ex_lib_opts.cufft = true;
        if (std.mem.indexOf(u8, ex.libs, "cupti") != null) ex_lib_opts.cupti = true;
        if (std.mem.indexOf(u8, ex.libs, "cufile") != null) ex_lib_opts.cufile = true;
        if (std.mem.indexOf(u8, ex.libs, "nvtx") != null) ex_lib_opts.nvtx = true;
        linkLibraries(example, ex_lib_opts);


        const install_example = b.addInstallArtifact(example, .{});
        const example_step = b.step(
            try std.fmt.allocPrint(b.allocator, "example-{s}", .{step_name}),
            try std.fmt.allocPrint(b.allocator, "Build {s} example", .{ex.name}),
        );
        example_step.dependOn(&install_example.step);

        const run_example = b.addRunArtifact(example);
        const run_example_step = b.step(
            try std.fmt.allocPrint(b.allocator, "run-{s}", .{step_name}),
            try std.fmt.allocPrint(b.allocator, "Run {s}: {s}", .{ ex.name, ex.desc }),
        );
        run_example_step.dependOn(&run_example.step);
    }

    // ── GPU Kernel Compilation (Zig → PTX) ──
    // Compiles .zig kernel files to PTX assembly using the nvptx64-cuda-none target.
    // Usage: zig build compile-kernels
    //        zig build compile-kernels -Dgpu-arch=sm_80
    //        zig build -Dembed-ptx        (embed PTX in binary for single-file deployment)
    const kernel_step = b.step("compile-kernels", "Compile Zig GPU kernels to PTX");

    // PTX embedding option for production deployment
    const embed_ptx = b.option(bool, "embed-ptx", "Embed PTX data in binary for single-file deployment (default: false)") orelse false;

    // GPU architecture option (default: sm_80 / Ampere)
    const gpu_arch_str = b.option([]const u8, "gpu-arch", "Target GPU SM architecture (default: sm_80)") orelse "sm_80";
    const sm_version = build_helpers.parseSmVersion(gpu_arch_str);

    // Resolve the nvptx64 target with the correct SM architecture
    const nvptx_target = build_helpers.resolveNvptxTarget(b, gpu_arch_str);

    // Build options module to pass SM version as comptime value
    const device_options = b.addOptions();
    device_options.addOption(u32, "sm_version", sm_version);

    // Device intrinsics module for GPU kernels
    const device_mod = b.createModule(.{
        .root_source_file = b.path("src/kernel/device.zig"),
        .target = nvptx_target,
        .optimize = .ReleaseFast,
    });
    device_mod.addOptions("build_options", device_options);

    // ── Kernel discovery + bridge generation via build_helpers.zig ──
    // Recursively scans kernel directory for .zig files containing `export fn`,
    // compiles each to PTX, and generates type-safe bridge modules.

    const kernel_dir = b.option([]const u8, "kernel-dir", "Root directory for kernel auto-discovery (default: src/kernel/)") orelse "src/kernel/";
    const discovered = build_helpers.discoverKernels(b, kernel_dir);
    const bridge_result = build_helpers.addBridgeModules(b, discovered, .{
        .embed_ptx = embed_ptx,
        .zcuda_bridge_mod = zcuda_bridge_mod,
        .zcuda_mod = zcuda_mod,
        .device_mod = device_mod,
        .nvptx_target = nvptx_target,
        .kernel_step = kernel_step,
        .target = target,
        .optimize = optimize,
    });

    // Discover example kernels for integration examples (if examples/kernel/ exists and
    // kernel-dir doesn't already point there to avoid duplicate discovery)
    const example_bridge_result = if (!std.mem.startsWith(u8, kernel_dir, "examples/kernel"))
        build_helpers.addBridgeModules(b, build_helpers.discoverKernels(b, "examples/kernel/"), .{
            .embed_ptx = embed_ptx,
            .zcuda_bridge_mod = zcuda_bridge_mod,
            .zcuda_mod = zcuda_mod,
            .device_mod = device_mod,
            .nvptx_target = nvptx_target,
            .kernel_step = kernel_step,
            .target = target,
            .optimize = optimize,
        })
    else
        bridge_result;

    // Configure CUDA paths on bridge modules (needed for host-side driver API calls)
    for (bridge_result.modules) |entry| {
        if (cuda_path) |cp| {
            configurePaths(entry.module, cp, cuda_include.?, cudnn_include, &lib_paths, b.allocator) catch continue;
        }
        linkLibrariesToModule(entry.module, lib_opts);
    }
    if (!std.mem.startsWith(u8, kernel_dir, "examples/kernel")) {
        for (example_bridge_result.modules) |entry| {
            if (cuda_path) |cp| {
                configurePaths(entry.module, cp, cuda_include.?, cudnn_include, &lib_paths, b.allocator) catch continue;
            }
            linkLibrariesToModule(entry.module, lib_opts);
        }
    }

    // ── 10_Integration host-side examples ──
    // Each integration example is an executable that imports zcuda + kernel bridge modules.
    const IntExample = struct {
        name: []const u8,
        desc: []const u8,
        path: []const u8,
        bridges: []const []const u8, // which bridge module import names to add
        libs: []const u8,
    };
    const integration_examples = [_]IntExample{
        .{ .name = "integration-module-load-launch", .desc = "Driver lifecycle: load+launch", .path = "examples/kernel/10_Integration/A_DriverLifecycle/module_load_launch.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        .{ .name = "integration-ptx-compile-execute", .desc = "PTX compile+execute", .path = "examples/kernel/10_Integration/A_DriverLifecycle/ptx_compile_execute.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        // B_StreamsAndEvents
        .{ .name = "integration-stream-callback", .desc = "Stream callback pattern", .path = "examples/kernel/10_Integration/B_StreamsAndEvents/stream_callback.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        .{ .name = "integration-stream-concurrency", .desc = "Multi-stream concurrency", .path = "examples/kernel/10_Integration/B_StreamsAndEvents/stream_concurrency.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        // C_CudaGraphs
        .{ .name = "integration-basic-graph", .desc = "CUDA Graph basics", .path = "examples/kernel/10_Integration/C_CudaGraphs/basic_graph.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        .{ .name = "integration-graph-replay-update", .desc = "Graph replay + update", .path = "examples/kernel/10_Integration/C_CudaGraphs/graph_replay_update.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        .{ .name = "integration-graph-with-deps", .desc = "Graph with dependencies", .path = "examples/kernel/10_Integration/C_CudaGraphs/graph_with_dependencies.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        // D_cuBLAS_Pipelines
        .{ .name = "integration-scale-bias-gemm", .desc = "cuBLAS Scale+Bias→GEMM→ReLU", .path = "examples/kernel/10_Integration/D_cuBLAS_Pipelines/scale_bias_gemm.zig", .bridges = &.{ "kernel_scale_bias", "kernel_relu" }, .libs = "cublas" },
        .{ .name = "integration-residual-gemm", .desc = "cuBLAS Residual GEMM", .path = "examples/kernel/10_Integration/D_cuBLAS_Pipelines/residual_gemm.zig", .bridges = &.{"kernel_residual_norm"}, .libs = "cublas" },
        // E_ErrorHandling
        .{ .name = "integration-error-recovery", .desc = "Error recovery patterns", .path = "examples/kernel/10_Integration/E_ErrorHandling/error_recovery.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        .{ .name = "integration-oob-launch", .desc = "Out-of-bounds launch", .path = "examples/kernel/10_Integration/E_ErrorHandling/oob_launch.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        // E_cuFFT_Pipelines
        .{ .name = "integration-fft-filter", .desc = "FFT filter pipeline", .path = "examples/kernel/10_Integration/E_cuFFT_Pipelines/fft_filter_pipeline.zig", .bridges = &.{ "kernel_signal_gen", "kernel_freq_filter" }, .libs = "cufft" },
        .{ .name = "integration-conv2d-fft", .desc = "2D convolution via FFT", .path = "examples/kernel/10_Integration/E_cuFFT_Pipelines/conv2d_fft.zig", .bridges = &.{ "kernel_pad_2d", "kernel_complex_mul" }, .libs = "cufft" },
        // F_Profiling
        .{ .name = "integration-occupancy-calc", .desc = "Occupancy calculator", .path = "examples/kernel/10_Integration/F_Profiling/occupancy_calculator.zig", .bridges = &.{"kernel_vector_add"}, .libs = "" },
        // F_cuRAND_Applications
        .{ .name = "integration-monte-carlo-option", .desc = "Monte Carlo option pricing", .path = "examples/kernel/10_Integration/F_cuRAND_Applications/monte_carlo_option.zig", .bridges = &.{"kernel_gbm_paths"}, .libs = "curand" },
        .{ .name = "integration-particle-system", .desc = "Particle system simulation", .path = "examples/kernel/10_Integration/F_cuRAND_Applications/particle_system.zig", .bridges = &.{ "kernel_particle_init", "kernel_particle_step" }, .libs = "curand" },
        // G_EndToEnd
        .{ .name = "integration-matmul-e2e", .desc = "Matmul end-to-end", .path = "examples/kernel/10_Integration/G_EndToEnd/matmul_e2e.zig", .bridges = &.{"kernel_matmul_naive"}, .libs = "" },
        .{ .name = "integration-reduction-e2e", .desc = "Reduction end-to-end", .path = "examples/kernel/10_Integration/G_EndToEnd/reduction_e2e.zig", .bridges = &.{"kernel_reduce_sum"}, .libs = "" },
        .{ .name = "integration-saxpy-e2e", .desc = "SAXPY end-to-end", .path = "examples/kernel/10_Integration/G_EndToEnd/saxpy_e2e.zig", .bridges = &.{"kernel_saxpy"}, .libs = "" },
        // G_LibraryCombined
        .{ .name = "integration-multi-library", .desc = "Multi-library pipeline", .path = "examples/kernel/10_Integration/G_LibraryCombined/multi_library_pipeline.zig", .bridges = &.{ "kernel_sigmoid", "kernel_extract_diag" }, .libs = "cublas,curand,cufft" },
        // H_TensorCore_Pipelines
        .{ .name = "integration-wmma-gemm-verify", .desc = "WMMA GEMM verification", .path = "examples/kernel/10_Integration/H_TensorCore_Pipelines/wmma_gemm_verify.zig", .bridges = &.{"kernel_wmma_gemm_f16"}, .libs = "cublas" },
        .{ .name = "integration-attention-pipeline", .desc = "Attention pipeline", .path = "examples/kernel/10_Integration/H_TensorCore_Pipelines/attention_pipeline.zig", .bridges = &.{ "kernel_wmma_gemm_bf16", "kernel_softmax" }, .libs = "cublas" },
        .{ .name = "integration-mixed-precision-train", .desc = "Mixed precision training", .path = "examples/kernel/10_Integration/H_TensorCore_Pipelines/mixed_precision_train.zig", .bridges = &.{ "kernel_wmma_gemm_f16", "kernel_relu" }, .libs = "cublas,curand" },
        // I_Performance
        .{ .name = "integration-perf-benchmark", .desc = "Zig kernel vs cuBLAS (Event-timed)", .path = "examples/kernel/10_Integration/I_Performance/perf_benchmark.zig", .bridges = &.{ "kernel_vector_add", "kernel_matmul_tiled" }, .libs = "cublas" },
    };

    const integration_step = b.step("example-integration", "Build 10_Integration examples");
    for (integration_examples) |iex| {
        const exe = b.addExecutable(.{
            .name = iex.name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(iex.path),
                .target = target,
                .optimize = optimize,
            }),
        });
        exe.root_module.addImport("zcuda", zcuda_mod);
        exe.root_module.addOptions("build_options", build_options);
        if (cuda_path) |cp| {
            try configurePaths(exe.root_module, cp, cuda_include.?, cudnn_include, &lib_paths, b.allocator);
        }

        // Link required libraries based on the libs field
        var int_lib_opts = lib_opts;
        if (std.mem.indexOf(u8, iex.libs, "cublas") != null) int_lib_opts.cublas = true;
        if (std.mem.indexOf(u8, iex.libs, "curand") != null) int_lib_opts.curand = true;
        if (std.mem.indexOf(u8, iex.libs, "cufft") != null) int_lib_opts.cufft = true;
        linkLibraries(exe, int_lib_opts);

        const install_exe = b.addInstallArtifact(exe, .{});

        // Add kernel bridge module imports (search both main and example bridges)
        for (iex.bridges) |bridge_name| {
            const mod = build_helpers.findBridge(bridge_result.modules, bridge_name) orelse
                build_helpers.findBridge(example_bridge_result.modules, bridge_name);
            if (mod) |m| {
                exe.root_module.addImport(bridge_name, m);
            }
            // For disk-mode bridges (LLVM-compiled, no embedded PTX), ensure the
            // PTX install step runs before the executable is installed.
            const ptx_step = build_helpers.findBridgeInstallStep(bridge_result.modules, bridge_name) orelse
                build_helpers.findBridgeInstallStep(example_bridge_result.modules, bridge_name);
            if (ptx_step) |s| {
                install_exe.step.dependOn(s);
            }
        }

        integration_step.dependOn(&install_exe.step);

        // Per-example build step: zig build example-integration-xxx
        const per_step = b.step(
            try std.fmt.allocPrint(b.allocator, "example-{s}", .{iex.name}),
            try std.fmt.allocPrint(b.allocator, "Build {s}", .{iex.desc}),
        );
        per_step.dependOn(&install_exe.step);
    }
}


fn addKernelTarget(
    b: *std.Build,
    kernel_step: *std.Build.Step,
    device_mod: *std.Build.Module,
    nvptx_target: std.Build.ResolvedTarget,
    name: []const u8,
    desc: []const u8,
    kernel_path: []const u8,
) void {
    const kernel_obj = b.addObject(.{
        .name = name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(kernel_path),
            .target = nvptx_target,
            .optimize = .ReleaseFast,
        }),
    });
    kernel_obj.root_module.addImport("zcuda_kernel", device_mod);

    const ptx_output = kernel_obj.getEmittedAsm();
    const install_ptx = b.addInstallFile(ptx_output, std.fmt.allocPrint(
        b.allocator,
        "bin/kernel/{s}.ptx",
        .{name},
    ) catch @panic("OOM"));

    kernel_step.dependOn(&install_ptx.step);

    const per_kernel_step = b.step(
        std.fmt.allocPrint(b.allocator, "kernel-{s}", .{name}) catch @panic("OOM"),
        std.fmt.allocPrint(b.allocator, "Compile {s}: {s}", .{ name, desc }) catch @panic("OOM"),
    );
    per_kernel_step.dependOn(&install_ptx.step);
}

/// Create a Run step for GPU test binaries that bypasses Zig's `--listen=-`
/// IPC protocol. CUDA's dynamically-linked driver library conflicts with the
/// test runner's binary stdout protocol, causing spurious failures.
///
/// This mimics `addRunArtifact` but skips `enableTestRunnerMode()`, so no
/// `--listen=-` argument is passed. The test binary runs normally and reports
/// results via exit code only (0 = pass, non-zero = fail).
fn addGpuTestRun(b: *std.Build, exe: *Step.Compile) *Step.Run {
    const run_step = Step.Run.create(b, b.fmt("run {s}", .{@tagName(exe.kind)}));
    run_step.producer = exe;
    run_step.addArtifactArg(exe);
    run_step.expectExitCode(0);
    return run_step;
}