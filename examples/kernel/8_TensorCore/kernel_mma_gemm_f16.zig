// examples/kernel/8_TensorCore/kernel_mma_gemm_f16.zig — MMA PTX f16→f32 (m16n8k16, sm_80+)
//
// Reference: PTX-level tensor core programming (no WMMA wrapper)
// API exercised: tensor_core.mma_f16_f32

const cuda = @import("zcuda_kernel");
const tc = cuda.tensor_core;

/// MMA PTX GEMM: lower-level than WMMA, outputs 16×8 tiles.
/// Each warp computes a m16n8k16 tile using direct MMA PTX.
export fn mmaGemmF16(
    A: [*]const u16,
    B: [*]const u16,
    C: [*]f32,
    D: [*]f32,
    M: u32,
    N: u32,
    K: u32,
) callconv(.kernel) void {
    const warp_row = (cuda.blockIdx().y * cuda.blockDim().y + cuda.threadIdx().y) / cuda.warpSize * 16;
    const warp_col = cuda.blockIdx().x * 8;

    if (warp_row >= M or warp_col >= N) return;

    // Initialize accumulator
    var acc: tc.MmaFragC_f32 = .{ 0.0, 0.0, 0.0, 0.0 };

    // Add C if present
    acc[0] += C[warp_row * N + warp_col];
    acc[1] += C[(warp_row + 8) * N + warp_col];
    acc[2] += C[warp_row * N + warp_col + 4];
    acc[3] += C[(warp_row + 8) * N + warp_col + 4];

    var k: u32 = 0;
    while (k < K) : (k += 16) {
        const a_frag: tc.MmaFragA_f16 = .{
            @bitCast([2]u16{ @intCast(A[warp_row * K + k]), @intCast(A[warp_row * K + k + 1]) }),
            @bitCast([2]u16{ @intCast(A[(warp_row + 8) * K + k]), @intCast(A[(warp_row + 8) * K + k + 1]) }),
            @bitCast([2]u16{ @intCast(A[warp_row * K + k + 8]), @intCast(A[warp_row * K + k + 9]) }),
            @bitCast([2]u16{ @intCast(A[(warp_row + 8) * K + k + 8]), @intCast(A[(warp_row + 8) * K + k + 9]) }),
        };

        const b_frag: tc.MmaFragB_f16 = .{
            @bitCast([2]u16{ @intCast(B[k * N + warp_col]), @intCast(B[(k + 1) * N + warp_col]) }),
            @bitCast([2]u16{ @intCast(B[(k + 8) * N + warp_col]), @intCast(B[(k + 9) * N + warp_col]) }),
        };

        acc = tc.mma_f16_f32(a_frag, b_frag, acc);
    }

    D[warp_row * N + warp_col] = acc[0];
    D[(warp_row + 8) * N + warp_col] = acc[1];
    D[warp_row * N + warp_col + 4] = acc[2];
    D[(warp_row + 8) * N + warp_col + 4] = acc[3];
}
