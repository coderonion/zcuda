// examples/kernel/8_TensorCore/kernel_wmma_gemm_tf32.zig — WMMA tf32→f32 GEMM (sm_80+)
//
// Reference: cuda-samples/3_CUDA_Features/tf32TensorCoreGemm
// API exercised: tensor_core.mma_tf32_f32

const cuda = @import("zcuda_kernel");
const tc = cuda.tensor_core;

/// TF32 GEMM using MMA PTX (m16n8k8, tf32→f32).
/// TF32 provides f32-like range with reduced mantissa for higher throughput.
export fn wmmaGemmTF32(
    A: [*]const f32, // tf32 stored as f32 (hardware truncates mantissa)
    B: [*]const f32,
    C: [*]f32,
    D: [*]f32,
    M: u32,
    N: u32,
    K: u32,
) callconv(.kernel) void {
    const warp_row = (cuda.blockIdx().y * cuda.blockDim().y + cuda.threadIdx().y) / cuda.warpSize * 16;
    const warp_col = cuda.blockIdx().x * 8;

    if (warp_row >= M or warp_col >= N) return;

    var acc: tc.MmaFragC_f32 = .{
        C[warp_row * N + warp_col],
        C[(warp_row + 8) * N + warp_col],
        C[warp_row * N + warp_col + 4],
        C[(warp_row + 8) * N + warp_col + 4],
    };

    var k: u32 = 0;
    while (k < K) : (k += 8) {
        // TF32 fragments — stored as u32 (bit-cast from f32)
        const a_frag: tc.MmaFragA_tf32 = .{
            @bitCast(A[warp_row * K + k]),
            @bitCast(A[(warp_row + 8) * K + k]),
            @bitCast(A[warp_row * K + k + 4]),
            @bitCast(A[(warp_row + 8) * K + k + 4]),
        };

        const b_frag: tc.MmaFragB_tf32 = .{
            @bitCast(B[k * N + warp_col]),
            @bitCast(B[(k + 4) * N + warp_col]),
        };

        acc = tc.mma_tf32_f32(a_frag, b_frag, acc);
    }

    D[warp_row * N + warp_col] = acc[0];
    D[(warp_row + 8) * N + warp_col] = acc[1];
    D[warp_row * N + warp_col + 4] = acc[2];
    D[(warp_row + 8) * N + warp_col + 4] = acc[3];
}
