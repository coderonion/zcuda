// kernels/softmax.zig — Numerically stable softmax using shared memory + warp shuffle
//
// Features: SharedArray, warp shuffle, two-pass reduction, grid-stride loop
//
// Algorithm (per row):
//   1. Find max value (for numerical stability)
//   2. Compute sum of exp(x - max)
//   3. Normalize: output[i] = exp(x[i] - max) / sum
//
// Each block processes one row of the input matrix [rows × cols].

const cuda = @import("zcuda_kernel");
const smem = cuda.shared_mem;

const BLOCK_SIZE = 256;

export fn softmax(
    input: [*]const f32,
    output: [*]f32,
    rows: u32,
    cols: u32,
) callconv(.kernel) void {
    const row = cuda.blockIdx().x;
    if (row >= rows) return;

    const tid = cuda.threadIdx().x;
    const row_offset = row * cols;

    const sdata = smem.SharedArray(f32, BLOCK_SIZE);
    const s = sdata.ptr();

    // ── Pass 1: Find max value in row ──
    var max_val: f32 = -3.40282347e+38; // -FLT_MAX
    var i = tid;
    while (i < cols) : (i += BLOCK_SIZE) {
        max_val = cuda.fmaxf(max_val, input[row_offset + i]);
    }

    // Warp reduction for max
    max_val = cuda.fmaxf(max_val, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(max_val), 16, 32)));
    max_val = cuda.fmaxf(max_val, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(max_val), 8, 32)));
    max_val = cuda.fmaxf(max_val, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(max_val), 4, 32)));
    max_val = cuda.fmaxf(max_val, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(max_val), 2, 32)));
    max_val = cuda.fmaxf(max_val, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(max_val), 1, 32)));

    // Block reduction via shared memory
    if (tid % cuda.warpSize == 0) {
        s[tid / cuda.warpSize] = max_val;
    }
    cuda.__syncthreads();

    if (tid < cuda.warpSize) {
        max_val = if (tid < (BLOCK_SIZE / cuda.warpSize)) s[tid] else -3.40282347e+38;
        max_val = cuda.fmaxf(max_val, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(max_val), 4, 32)));
        max_val = cuda.fmaxf(max_val, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(max_val), 2, 32)));
        max_val = cuda.fmaxf(max_val, @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(max_val), 1, 32)));
    }
    if (tid == 0) s[0] = max_val;
    cuda.__syncthreads();
    max_val = s[0];

    // ── Pass 2: Compute sum of exp(x - max) ──
    var exp_sum: f32 = 0.0;
    i = tid;
    while (i < cols) : (i += BLOCK_SIZE) {
        exp_sum += cuda.__expf(input[row_offset + i] - max_val);
    }

    // Warp + block reduction for sum
    exp_sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(exp_sum), 16, 32));
    exp_sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(exp_sum), 8, 32));
    exp_sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(exp_sum), 4, 32));
    exp_sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(exp_sum), 2, 32));
    exp_sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(exp_sum), 1, 32));

    if (tid % cuda.warpSize == 0) {
        s[tid / cuda.warpSize] = exp_sum;
    }
    cuda.__syncthreads();

    if (tid < cuda.warpSize) {
        exp_sum = if (tid < (BLOCK_SIZE / cuda.warpSize)) s[tid] else 0.0;
        exp_sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(exp_sum), 4, 32));
        exp_sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(exp_sum), 2, 32));
        exp_sum += @bitCast(cuda.__shfl_down_sync(cuda.FULL_MASK, @bitCast(exp_sum), 1, 32));
    }
    if (tid == 0) s[0] = exp_sum;
    cuda.__syncthreads();

    const inv_sum = cuda.rsqrtf(s[0] * s[0]); // idiom for 1/s[0] via rsqrt(x²)
    // Actually: 1/sum
    const sum_val = s[0];

    // ── Pass 3: Normalize ──
    i = tid;
    while (i < cols) : (i += BLOCK_SIZE) {
        output[row_offset + i] = cuda.__expf(input[row_offset + i] - max_val) / sum_val;
    }
    _ = inv_sum;
}
