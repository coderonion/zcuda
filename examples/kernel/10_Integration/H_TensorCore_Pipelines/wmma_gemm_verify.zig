/// TensorCore Integration: WMMA f16 GEMM vs cuBLAS Verification
///
/// Pipeline:
///   1. Custom WMMA f16→f32 Zig kernel (from 8_TensorCore/kernel_wmma_gemm_f16.zig)
///   2. cuBLAS HGEMM on same data
///   3. CPU reference (f64 precision)
///   4. Cross-compare all three results
///
/// Demonstrates:
///   - Loading and launching a TensorCore Zig kernel
///   - Using cuBLAS as a correctness oracle
///   - Multi-reference verification strategy
///
/// Reference: cuda-samples/cudaTensorCoreGemm
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
// Bridge provides: .load(), .getFunction(mod, .enum), .source_path, .source
const std = @import("std");
const cuda = @import("zcuda");

const kernel_wmma_gemm_f16 = @import("kernel_wmma_gemm_f16");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== TensorCore WMMA f16 GEMM vs cuBLAS Verification ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // ── Load custom WMMA kernel ──
    const mod = try kernel_wmma_gemm_f16.load(ctx, std.heap.page_allocator);
    defer mod.deinit();
    const wmma_fn = try kernel_wmma_gemm_f16.getFunction(mod, .wmmaGemmF16);

    // Tile-aligned dimensions (multiples of 16)
    const M: u32 = 64;
    const N: u32 = 64;
    const K: u32 = 64;

    // ── Prepare f16 data (packed as u16) ──
    var h_A_f16: [M * K]u16 = undefined;
    var h_B_f16: [K * N]u16 = undefined;
    var h_A_f32: [M * K]f32 = undefined;
    var h_B_f32: [K * N]f32 = undefined;

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();

    for (0..M * K) |i| {
        const val: f32 = (@as(f32, @floatFromInt(random.intRangeAtMost(i32, -10, 10)))) / 10.0;
        h_A_f32[i] = val;
        h_A_f16[i] = @bitCast(@as(f16, @floatCast(val)));
    }
    for (0..K * N) |i| {
        const val: f32 = (@as(f32, @floatFromInt(random.intRangeAtMost(i32, -10, 10)))) / 10.0;
        h_B_f32[i] = val;
        h_B_f16[i] = @bitCast(@as(f16, @floatCast(val)));
    }

    // ── Method 1: Custom WMMA Zig Kernel ──
    const d_A_f16 = try stream.cloneHtoD(u16, &h_A_f16);
    defer d_A_f16.deinit();
    const d_B_f16 = try stream.cloneHtoD(u16, &h_B_f16);
    defer d_B_f16.deinit();
    const d_C_zeros = try stream.allocZeros(f32, allocator, M * N);
    defer d_C_zeros.deinit();
    const d_D_wmma = try stream.allocZeros(f32, allocator, M * N);
    defer d_D_wmma.deinit();

    // Launch WMMA kernel: 1 warp per 16×16 tile, need (M/16)×(N/16) warps
    const grid_x = N / 16;
    const grid_y = M / 16;
    try stream.launch(wmma_fn, .{
        .grid_dim = .{ .x = grid_x, .y = grid_y },
        .block_dim = .{ .x = 32 },
    }, .{
        d_A_f16.devicePtr(), d_B_f16.devicePtr(), d_C_zeros.devicePtr(), d_D_wmma.devicePtr(),
        M,                   N,                   K,
    });
    std.debug.print("✓ WMMA kernel launched ({d}×{d} grid, 32 threads/block)\n", .{ grid_x, grid_y });

    // ── Method 2: cuBLAS SGEMM (f32 reference) ──
    const d_A_f32 = try stream.cloneHtoD(f32, &h_A_f32);
    defer d_A_f32.deinit();
    const d_B_f32 = try stream.cloneHtoD(f32, &h_B_f32);
    defer d_B_f32.deinit();
    const d_D_cublas = try stream.allocZeros(f32, allocator, M * N);
    defer d_D_cublas.deinit();

    // cuBLAS uses column-major; for row-major C = A × B, compute C^T = B^T × A^T
    try blas.sgemm(.no_transpose, .no_transpose, @intCast(N), @intCast(M), @intCast(K), 1.0, d_B_f32, @intCast(N), d_A_f32, @intCast(K), 0.0, d_D_cublas, @intCast(N));
    std.debug.print("✓ cuBLAS SGEMM completed\n", .{});

    // ── Method 3: CPU reference (f64 precision) ──
    var h_D_cpu: [M * N]f64 = undefined;
    for (0..M) |r| {
        for (0..N) |c| {
            var sum: f64 = 0.0;
            for (0..K) |kk| {
                const a: f64 = @floatCast(h_A_f32[r * K + kk]);
                const b: f64 = @floatCast(h_B_f32[kk * N + c]);
                sum += a * b;
            }
            h_D_cpu[r * N + c] = sum;
        }
    }
    std.debug.print("✓ CPU f64 reference computed\n\n", .{});

    // ── Cross-compare ──
    var h_D_wmma: [M * N]f32 = undefined;
    var h_D_blas: [M * N]f32 = undefined;
    try stream.memcpyDtoH(f32, &h_D_wmma, d_D_wmma);
    try stream.memcpyDtoH(f32, &h_D_blas, d_D_cublas);

    var max_err_wmma_cpu: f64 = 0.0;
    var max_err_blas_cpu: f64 = 0.0;
    var max_err_wmma_blas: f32 = 0.0;

    for (0..M * N) |i| {
        max_err_wmma_cpu = @max(max_err_wmma_cpu, @abs(@as(f64, h_D_wmma[i]) - h_D_cpu[i]));
        max_err_blas_cpu = @max(max_err_blas_cpu, @abs(@as(f64, h_D_blas[i]) - h_D_cpu[i]));
        max_err_wmma_blas = @max(max_err_wmma_blas, @abs(h_D_wmma[i] - h_D_blas[i]));
    }

    std.debug.print("── Verification Results ──\n", .{});
    std.debug.print("  WMMA  vs CPU(f64):   max error = {e:.6}\n", .{max_err_wmma_cpu});
    std.debug.print("  cuBLAS vs CPU(f64):  max error = {e:.6}\n", .{max_err_blas_cpu});
    std.debug.print("  WMMA  vs cuBLAS:     max error = {e:.6}\n\n", .{@as(f64, max_err_wmma_blas)});

    // f16 GEMM should be within ~1e-1 of f64 reference (f16 has limited precision)
    const wmma_tol: f64 = 0.5; // f16 precision is ~3 decimal digits
    const blas_tol: f64 = 1e-4; // f32 SGEMM should be very close to f64

    if (max_err_wmma_cpu > wmma_tol) {
        std.debug.print("✗ FAIL: WMMA kernel error too large\n", .{});
        return error.WmmaVerificationFailed;
    }
    if (max_err_blas_cpu > blas_tol) {
        std.debug.print("✗ FAIL: cuBLAS error too large\n", .{});
        return error.CublasVerificationFailed;
    }
    std.debug.print("✓ All three methods agree within tolerance\n", .{});
    std.debug.print("  (f16 WMMA within {d:.1}, f32 cuBLAS within {e})\n", .{ wmma_tol, blas_tol });
}
