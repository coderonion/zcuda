/// Performance Benchmark: Zig Kernel vs cuBLAS — Event-timed comparison
///
/// Measures and compares:
///   1. Memory bandwidth  — HtoD and DtoH (f32 arrays, 4 MB)
///   2. Zig vectorAdd kernel  — custom Zig kernel via bridge module
///   3. cuBLAS saxpy         — equivalent cuBLAS Level-1 operation
///   4. Zig kernel matmul    — tiled GEMM (512×512)
///   5. cuBLAS sgemm         — equivalent cuBLAS Level-3 operation
///
/// All timed runs use CudaEvent.elapsedTime() with a warm-up pass.
///
/// Reference: cuda-samples/benchmarkFunctionality, cuda-samples/matrixMul
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──
const std = @import("std");
const cuda = @import("zcuda");

const kernel_vector_add = @import("kernel_vector_add");
const kernel_matmul_tiled = @import("kernel_matmul_tiled");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== zcuda Performance Benchmark: Zig Kernel vs cuBLAS ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n", .{ctx.name()});

    const sm = try ctx.computeCapability();
    std.debug.print("Compute capability: {}.{}\n\n", .{ sm.major, sm.minor });

    const stream = ctx.defaultStream();

    // ── CUDA events ──
    const start = try stream.createEvent(0);
    defer start.deinit();
    const stop = try stream.createEvent(0);
    defer stop.deinit();

    // ── cuBLAS ──
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // ── Load kernel bridge modules ──
    const mod_va = try kernel_vector_add.load(ctx, allocator);
    defer mod_va.deinit();
    const va_fn = try kernel_vector_add.getFunction(mod_va, .vectorAdd);

    const mod_mm = try kernel_matmul_tiled.load(ctx, allocator);
    defer mod_mm.deinit();
    const mm_fn = try kernel_matmul_tiled.getFunction(mod_mm, .tiled_matmul);

    // ────────────────────────────────────────────────────────────
    // Benchmark 1: Memory Bandwidth  (4 MB = 1M f32)
    // ────────────────────────────────────────────────────────────
    const vec_n: usize = 1024 * 1024;
    const size_bytes = vec_n * @sizeOf(f32);
    const size_mb: f64 = @as(f64, @floatFromInt(size_bytes)) / (1024.0 * 1024.0);

    std.debug.print("─── Benchmark 1: Memory Bandwidth ({d:.0} MB) ───\n", .{size_mb});

    const h_A = try allocator.alloc(f32, vec_n);
    defer allocator.free(h_A);
    const h_B = try allocator.alloc(f32, vec_n);
    defer allocator.free(h_B);
    const h_out = try allocator.alloc(f32, vec_n);
    defer allocator.free(h_out);
    for (h_A, 0..) |*v, i| v.* = @floatFromInt(i % 1000);
    for (h_B, 0..) |*v, i| v.* = @floatFromInt((i + 1) % 1000);

    // HtoD: transfer both arrays
    try start.record(stream);
    const d_A = try stream.cloneHtoD(f32, h_A);
    defer d_A.deinit();
    const d_B = try stream.cloneHtoD(f32, h_B);
    defer d_B.deinit();
    try stop.record(stream);
    try stop.synchronize();
    const htod_ms: f64 = @floatCast(try start.elapsedTime(stop));
    const htod_gb = (size_mb * 2.0) / 1024.0 / (htod_ms / 1000.0);

    // DtoH
    try start.record(stream);
    try stream.memcpyDtoH(f32, h_out, d_A);
    try stop.record(stream);
    try stop.synchronize();
    const dtoh_ms: f64 = @floatCast(try start.elapsedTime(stop));
    const dtoh_gb = size_mb / 1024.0 / (dtoh_ms / 1000.0);

    std.debug.print("  HtoD (2×{d:.0} MB): {d:.3} ms  → {d:.2} GB/s\n", .{ size_mb, htod_ms, htod_gb });
    std.debug.print("  DtoH ({d:.0} MB):   {d:.3} ms  → {d:.2} GB/s\n\n", .{ size_mb, dtoh_ms, dtoh_gb });

    // ────────────────────────────────────────────────────────────
    // Benchmark 2: vectorAdd (Zig kernel) vs saxpy (cuBLAS)
    // ────────────────────────────────────────────────────────────
    std.debug.print("─── Benchmark 2: vectorAdd / saxpy  (N = {}) ───\n", .{vec_n});

    const d_out = try stream.alloc(f32, allocator, vec_n);
    defer d_out.deinit();

    const va_config = cuda.LaunchConfig.forNumElems(@intCast(vec_n));
    const va_n: u32 = @intCast(vec_n);

    // Zig vectorAdd warm-up + timed run
    try stream.launch(va_fn, va_config, .{ &d_A, &d_B, &d_out, va_n });
    try stream.synchronize();
    try start.record(stream);
    try stream.launch(va_fn, va_config, .{ &d_A, &d_B, &d_out, va_n });
    try stop.record(stream);
    try stop.synchronize();
    const zig_va_ms: f64 = @floatCast(try start.elapsedTime(stop));
    const zig_va_gops = @as(f64, @floatFromInt(vec_n)) / 1.0e9 / (zig_va_ms / 1000.0);

    // cuBLAS saxpy warm-up + timed run (y = 1*x + y)
    const va_n_i32: i32 = @intCast(vec_n);
    try blas.saxpy(va_n_i32, 1.0, d_A, d_B);
    try stream.synchronize();
    try start.record(stream);
    try blas.saxpy(va_n_i32, 1.0, d_A, d_B);
    try stop.record(stream);
    try stop.synchronize();
    const blas_va_ms: f64 = @floatCast(try start.elapsedTime(stop));
    const blas_va_gops = @as(f64, @floatFromInt(vec_n)) / 1.0e9 / (blas_va_ms / 1000.0);

    std.debug.print("  Zig kernel vectorAdd: {d:.3} ms  → {d:.2} GElem/s\n", .{ zig_va_ms, zig_va_gops });
    std.debug.print("  cuBLAS saxpy:         {d:.3} ms  → {d:.2} GElem/s\n", .{ blas_va_ms, blas_va_gops });
    std.debug.print("  Ratio (Zig/cuBLAS):   {d:.2}x\n\n", .{zig_va_ms / blas_va_ms});

    // ────────────────────────────────────────────────────────────
    // Benchmark 3: tiled_matmul (Zig kernel) vs sgemm (cuBLAS)
    // ────────────────────────────────────────────────────────────
    const mat_n: usize = 512;
    const mat_nn = mat_n * mat_n;
    std.debug.print("─── Benchmark 3: matmul / sgemm  ({}×{}) ───\n", .{ mat_n, mat_n });

    const d_mA = try stream.alloc(f32, allocator, mat_nn);
    defer d_mA.deinit();
    const d_mB = try stream.alloc(f32, allocator, mat_nn);
    defer d_mB.deinit();
    const d_mC = try stream.allocZeros(f32, allocator, mat_nn);
    defer d_mC.deinit();

    // init d_mA, d_mB using vectorAdd kernel (reuse)
    const init_config = cuda.LaunchConfig.forNumElems(@intCast(mat_nn));
    const mat_nn_u32: u32 = @intCast(mat_nn);
    try stream.launch(va_fn, init_config, .{ &d_mA, &d_mA, &d_mB, mat_nn_u32 }); // B = A+A
    try stream.synchronize();

    const mm_n: u32 = @intCast(mat_n);
    const mm_config = cuda.LaunchConfig{
        .grid_dim = .{
            .x = @intCast((mat_n + 15) / 16),
            .y = @intCast((mat_n + 15) / 16),
            .z = 1,
        },
        .block_dim = .{ .x = 16, .y = 16, .z = 1 },
    };

    // Zig tiled_matmul warm-up + timed run
    try stream.launch(mm_fn, mm_config, .{ &d_mA, &d_mB, &d_mC, mm_n, mm_n, mm_n });
    try stream.synchronize();
    try start.record(stream);
    try stream.launch(mm_fn, mm_config, .{ &d_mA, &d_mB, &d_mC, mm_n, mm_n, mm_n });
    try stop.record(stream);
    try stop.synchronize();
    const zig_mm_ms: f64 = @floatCast(try start.elapsedTime(stop));
    const zig_mm_gflops = 2.0 * @as(f64, @floatFromInt(mat_n * mat_n * mat_n)) / 1.0e9 / (zig_mm_ms / 1000.0);

    // cuBLAS sgemm warm-up + timed run
    const mm_n_i32: i32 = @intCast(mat_n);
    try blas.sgemm(.no_transpose, .no_transpose, mm_n_i32, mm_n_i32, mm_n_i32, 1.0, d_mA, mm_n_i32, d_mB, mm_n_i32, 0.0, d_mC, mm_n_i32);
    try stream.synchronize();
    try start.record(stream);
    try blas.sgemm(.no_transpose, .no_transpose, mm_n_i32, mm_n_i32, mm_n_i32, 1.0, d_mA, mm_n_i32, d_mB, mm_n_i32, 0.0, d_mC, mm_n_i32);
    try stop.record(stream);
    try stop.synchronize();
    const blas_mm_ms: f64 = @floatCast(try start.elapsedTime(stop));
    const blas_mm_gflops = 2.0 * @as(f64, @floatFromInt(mat_n * mat_n * mat_n)) / 1.0e9 / (blas_mm_ms / 1000.0);

    std.debug.print("  Zig kernel tiled_matmul: {d:.3} ms  → {d:.2} GFLOP/s\n", .{ zig_mm_ms, zig_mm_gflops });
    std.debug.print("  cuBLAS sgemm:            {d:.3} ms  → {d:.2} GFLOP/s\n", .{ blas_mm_ms, blas_mm_gflops });
    std.debug.print("  Ratio (Zig/cuBLAS):      {d:.2}x\n\n", .{zig_mm_ms / blas_mm_ms});

    // ── Summary ──────────────────────────────────────────────────────────────
    std.debug.print("━━━ Summary ({s}) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{ctx.name()});
    std.debug.print("  Memory HtoD (2×{d:.0} MB):      {d:.2} GB/s\n", .{ size_mb, htod_gb });
    std.debug.print("  Memory DtoH ({d:.0} MB):         {d:.2} GB/s\n", .{ size_mb, dtoh_gb });
    std.debug.print("  Zig vectorAdd (N={}):  {d:.3} ms  → {d:.2} GElem/s\n", .{ vec_n, zig_va_ms, zig_va_gops });
    std.debug.print("  cuBLAS saxpy  (N={}):  {d:.3} ms  → {d:.2} GElem/s\n", .{ vec_n, blas_va_ms, blas_va_gops });
    std.debug.print("  Zig tiled_matmul ({}²): {d:.3} ms  → {d:.2} GFLOP/s\n", .{ mat_n, zig_mm_ms, zig_mm_gflops });
    std.debug.print("  cuBLAS sgemm     ({}²): {d:.3} ms  → {d:.2} GFLOP/s\n", .{ mat_n, blas_mm_ms, blas_mm_gflops });
    std.debug.print("\n✓ Benchmark complete\n", .{});
}
