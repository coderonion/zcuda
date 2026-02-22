// kernels/tiled_matmul.zig — Tiled matrix multiplication using shared memory
//
// Features: SharedArray, 2D thread blocks, loop tiling, __syncthreads
//
// Algorithm:
//   C[m×n] = A[m×k] × B[k×n]
//   Each thread block computes a TILE×TILE sub-matrix of C by iterating
//   over tiles of A and B, loading each tile into shared memory.
//
// This is the canonical GPU programming example demonstrating the
// performance benefit of shared memory (reducing global memory traffic
// from O(n³) to O(n³/TILE)).

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const TILE = 16;

/// Tiled matrix multiplication: C = A * B
export fn tiled_matmul(
    a: [*]const f32,
    b: [*]const f32,
    c: [*]f32,
    m: u32,
    n: u32,
    k: u32,
) callconv(.kernel) void {
    // Allocate a single shared memory block for both tiles to avoid
    // Zig comptime memoization aliasing (two SharedArray calls with
    // identical type+size return the same pointer).
    const tiles = smem.SharedArray(f32, TILE * TILE * 2);
    const sa = tiles.ptr()[0 .. TILE * TILE];
    const sb = tiles.ptr()[TILE * TILE .. TILE * TILE * 2];

    const tx = cuda.threadIdx().x;
    const ty = cuda.threadIdx().y;
    const row = cuda.blockIdx().y * TILE + ty;
    const col = cuda.blockIdx().x * TILE + tx;

    var sum: f32 = 0.0;

    // Iterate over tiles along the K dimension
    var t: u32 = 0;
    while (t * TILE < k) : (t += 1) {
        // Cooperatively load tile of A into shared memory
        const a_col = t * TILE + tx;
        if (row < m and a_col < k) {
            sa[ty * TILE + tx] = a[row * k + a_col];
        } else {
            sa[ty * TILE + tx] = 0.0;
        }

        // Cooperatively load tile of B into shared memory
        const b_row = t * TILE + ty;
        if (b_row < k and col < n) {
            sb[ty * TILE + tx] = b[b_row * n + col];
        } else {
            sb[ty * TILE + tx] = 0.0;
        }

        cuda.__syncthreads();

        // Compute partial dot product from this tile
        for (0..TILE) |kk| {
            sum += sa[ty * TILE + kk] * sb[kk * TILE + tx];
        }

        cuda.__syncthreads();
    }

    // Write result
    if (row < m and col < n) {
        c[row * n + col] = sum;
    }
}
