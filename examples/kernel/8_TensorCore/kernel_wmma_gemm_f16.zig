// examples/kernel/8_TensorCore/kernel_wmma_gemm_f16.zig — WMMA f16→f32 GEMM (sm_70+)
//
// Reference: cuda-samples/3_CUDA_Features/cudaTensorCoreGemm
// API exercised: tensor_core.wmma_load_a_f16, wmma_load_b_f16, wmma_load_c_f32,
//                wmma_mma_f16_f32, wmma_store_d_f32

const cuda = @import("zcuda_kernel");
const tc = cuda.tensor_core;

const M_TILE = 16;
const N_TILE = 16;
const K_TILE = 16;

/// WMMA GEMM: D[m×n] = A[m×k] × B[k×n] + C[m×n]
/// Each warp computes one 16×16 output tile.
/// A is row-major f16, B is col-major f16, C/D are row-major f32.
export fn wmmaGemmF16(
    A: [*]const u16, // f16 packed as u16
    B: [*]const u16,
    C: [*]const f32,
    D: [*]f32,
    M: u32,
    N: u32,
    K: u32,
) callconv(.kernel) void {
    // Each warp handles one 16×16 output tile
    const warp_row = (cuda.blockIdx().y * cuda.blockDim().y + cuda.threadIdx().y) / cuda.warpSize * M_TILE;
    const warp_col = cuda.blockIdx().x * N_TILE;

    if (warp_row >= M or warp_col >= N) return;

    // Load accumulator C
    const c_frag = tc.wmma_load_c_f32(@ptrCast(@alignCast(&C[warp_row * N + warp_col])), N);

    var acc = c_frag;

    // Loop over K dimension in tiles of K_TILE
    var k: u32 = 0;
    while (k < K) : (k += K_TILE) {
        // Load A tile (row-major)
        const a_frag = tc.wmma_load_a_f16(@ptrCast(@alignCast(&A[warp_row * K + k])), K);

        // Load B tile (col-major)
        const b_frag = tc.wmma_load_b_f16(@ptrCast(@alignCast(&B[k * N + warp_col])), N);

        // D = A * B + C
        acc = tc.wmma_mma_f16_f32(a_frag, b_frag, acc);
    }

    // Store result
    tc.wmma_store_d_f32(@ptrCast(@alignCast(&D[warp_row * N + warp_col])), acc, N);
}
