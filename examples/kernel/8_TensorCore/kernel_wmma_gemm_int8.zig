// examples/kernel/8_TensorCore/kernel_wmma_gemm_int8.zig — WMMA s8→s32 integer GEMM (sm_75+)
//
// Reference: cuda-samples/3_CUDA_Features/immaTensorCoreGemm
// API exercised: tensor_core.wmma_mma_s8_s32

const cuda = @import("zcuda_kernel");
const tc = cuda.tensor_core;

/// Integer GEMM using WMMA (m8n8k16, s8→s32).
/// Used for quantized neural network inference (INT8).
export fn wmmaGemmInt8(
    A: [*]const i8,
    B: [*]const i8,
    C: [*]i32,
    D: [*]i32,
    M: u32,
    N: u32,
    K: u32,
) callconv(.kernel) void {
    const warp_row = (cuda.blockIdx().y * cuda.blockDim().y + cuda.threadIdx().y) / cuda.warpSize * 8;
    const warp_col = cuda.blockIdx().x * 8;

    if (warp_row >= M or warp_col >= N) return;

    // Load accumulator
    var acc: tc.WmmaFragC_s32 = .{
        C[warp_row * N + warp_col],
        C[(warp_row + 4) * N + warp_col],
    };

    // Tile over K in chunks of 16
    var k: u32 = 0;
    while (k < K) : (k += 16) {
        // Pack 4 i8 values into 1 u32 for fragment A
        const a_ptr: [*]const u32 = @ptrCast(@alignCast(&A[warp_row * K + k]));
        const a_frag: tc.WmmaFragA_s8 = .{a_ptr[0]};

        const b_ptr: [*]const u32 = @ptrCast(@alignCast(&B[k * N + warp_col]));
        const b_frag: tc.WmmaFragB_s8 = .{b_ptr[0]};

        acc = tc.wmma_mma_s8_s32(a_frag, b_frag, acc);
    }

    // Store
    D[warp_row * N + warp_col] = acc[0];
    D[(warp_row + 4) * N + warp_col] = acc[1];
}
