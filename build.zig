const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

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
    const cuda_path = try findCudaPath(b.allocator, cuda_path_opt);
    const cuda_include = try std.fmt.allocPrint(b.allocator, "{s}/include", .{cuda_path});
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
    build_options.addOption(bool, "enable_cudnn", enable_cudnn);
    build_options.addOption(bool, "enable_cusolver", enable_cusolver);
    build_options.addOption(bool, "enable_cusparse", enable_cusparse);
    build_options.addOption(bool, "enable_cufft", enable_cufft);
    build_options.addOption(bool, "enable_cupti", enable_cupti);
    build_options.addOption(bool, "enable_cufile", enable_cufile);
    build_options.addOption(bool, "enable_nvtx", enable_nvtx);

    // ── Export zcuda as a Zig package for downstream consumers ──
    // Create the zcuda module — downstream users just call:
    //   exe.root_module.addImport("zcuda", dep.module("zcuda"));
    // System library linking propagates transitively via the module.
    const zcuda_mod = b.addModule("zcuda", .{
        .root_source_file = b.path("src/cuda.zig"),
    });
    zcuda_mod.addOptions("build_options", build_options);
    try configurePaths(zcuda_mod, cuda_path, cuda_include, cudnn_include, &lib_paths, b.allocator);

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

    // Link CUDA libraries onto the exported module so they propagate
    // transitively to any downstream executable that imports "zcuda".
    linkLibrariesToModule(zcuda_mod, lib_opts);
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
        try configurePaths(unit.root_module, cuda_path, cuda_include, cudnn_include, &lib_paths, b.allocator);
        linkLibraries(unit, lib_opts);

        const run_unit = b.addRunArtifact(unit);
        unit_test_step.dependOn(&run_unit.step);
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
        try configurePaths(unit.root_module, cuda_path, cuda_include, cudnn_include, &lib_paths, b.allocator);
        linkLibraries(unit, lib_opts);

        const run_unit = b.addRunArtifact(unit);
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
        integration.root_module.addOptions("build_options", build_options);
        try configurePaths(integration.root_module, cuda_path, cuda_include, cudnn_include, &lib_paths, b.allocator);
        linkLibraries(integration, lib_opts);

        const run_integration = b.addRunArtifact(integration);
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
        try configurePaths(example.root_module, cuda_path, cuda_include, cudnn_include, &lib_paths, b.allocator);
        linkLibraries(example, lib_opts);

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
}
