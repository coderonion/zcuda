/// Kernel Attributes Example
///
/// Demonstrates querying kernel function attributes and occupancy:
/// 1. Query registers per thread
/// 2. Query shared memory, constant memory, local memory usage
/// 3. Query max threads per block
/// 4. PTX and binary version
///
/// Reference: cudarc/10-function-attributes + cuda-samples/simpleOccupancy
const std = @import("std");
const cuda = @import("zcuda");

const kernel_src =
    \\extern "C" __global__ void matmul_shared(
    \\    const float *A, const float *B, float *C,
    \\    int M, int N, int K
    \\) {
    \\    __shared__ float tile_A[16][16];
    \\    __shared__ float tile_B[16][16];
    \\
    \\    int row = blockIdx.y * 16 + threadIdx.y;
    \\    int col = blockIdx.x * 16 + threadIdx.x;
    \\    float sum = 0.0f;
    \\
    \\    for (int t = 0; t < (K + 15) / 16; t++) {
    \\        if (row < M && t * 16 + threadIdx.x < K)
    \\            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * 16 + threadIdx.x];
    \\        else
    \\            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
    \\
    \\        if (col < N && t * 16 + threadIdx.y < K)
    \\            tile_B[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
    \\        else
    \\            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
    \\
    \\        __syncthreads();
    \\
    \\        for (int i = 0; i < 16; i++)
    \\            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
    \\
    \\        __syncthreads();
    \\    }
    \\
    \\    if (row < M && col < N)
    \\        C[row * N + col] = sum;
    \\}
    \\
    \\extern "C" __global__ void simple_add(float *data, float val, int n) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) data[i] += val;
    \\}
;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Kernel Attributes Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n", .{ctx.name()});

    const cap = try ctx.computeCapability();
    std.debug.print("Compute Capability: {}.{}\n\n", .{ cap.major, cap.minor });

    // Compile kernels
    const ptx = try cuda.nvrtc.compilePtx(allocator, kernel_src);
    defer allocator.free(ptx);
    const module = try ctx.loadModule(ptx);
    defer module.deinit();

    const sys = cuda.driver.sys;

    // --- Query attributes for both kernels ---
    const kernels = [_]struct { name: []const u8, func: cuda.CudaFunction }{
        .{ .name = "matmul_shared", .func = try module.getFunction("matmul_shared") },
        .{ .name = "simple_add", .func = try module.getFunction("simple_add") },
    };

    for (&kernels) |k| {
        std.debug.print("━━━ Kernel: {s} ━━━\n", .{k.name});

        // Resource usage
        const num_regs = try cuda.driver.result.function.getAttribute(
            k.func.function,
            sys.CU_FUNC_ATTRIBUTE_NUM_REGS,
        );
        const shared_bytes = try cuda.driver.result.function.getAttribute(
            k.func.function,
            sys.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        );
        const const_bytes = try cuda.driver.result.function.getAttribute(
            k.func.function,
            sys.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
        );
        const local_bytes = try cuda.driver.result.function.getAttribute(
            k.func.function,
            sys.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        );

        std.debug.print("  Registers/thread:    {}\n", .{num_regs});
        std.debug.print("  Static shared mem:   {} bytes\n", .{shared_bytes});
        std.debug.print("  Constant mem:        {} bytes\n", .{const_bytes});
        std.debug.print("  Local mem/thread:    {} bytes\n", .{local_bytes});

        // Max threads
        const max_threads = try cuda.driver.result.function.getAttribute(
            k.func.function,
            sys.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        );
        std.debug.print("  Max threads/block:   {}\n", .{max_threads});

        // PTX/binary version
        const ptx_ver = try cuda.driver.result.function.getAttribute(
            k.func.function,
            sys.CU_FUNC_ATTRIBUTE_PTX_VERSION,
        );
        const bin_ver = try cuda.driver.result.function.getAttribute(
            k.func.function,
            sys.CU_FUNC_ATTRIBUTE_BINARY_VERSION,
        );
        std.debug.print("  PTX version:         {}.{}\n", .{ @divTrunc(ptx_ver, 10), @mod(ptx_ver, 10) });
        std.debug.print("  Binary version:      {}.{}\n", .{ @divTrunc(bin_ver, 10), @mod(bin_ver, 10) });

        std.debug.print("\n", .{});
    }

    std.debug.print("✓ Kernel attributes query complete\n", .{});
}
