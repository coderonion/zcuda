// examples/kernel/10_Integration/F_Profiling/occupancy_calculator.zig
// Reference: cuda-samples/0_Introduction/simpleOccupancy
// API: CudaFunction.maxActiveBlocksPerMultiprocessor, recommended block size
//
// ── Kernel Loading: Way 5 build.zig auto-generated bridge module ──
// Uses: @import("kernel_vector_add") — type-safe PTX loading via bridge module

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_vector_add = @import("kernel_vector_add");

/// Occupancy-based launch configuration.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();

    const module = try kernel_vector_add.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const func = try kernel_vector_add.getFunction(module, .vectorAdd);

    // Query optimal block size for maximum occupancy
    const optimal = try func.optimalBlockSize(.{ .shared_mem_bytes = 0 });
    const block_size = optimal.block_size;
    const min_grid_size = optimal.min_grid_size;

    // Query max active blocks for given block size
    const max_blocks = try func.maxActiveBlocksPerSM(block_size, 0);

    // Validate reasonable values
    if (block_size == 0 or block_size > 1024) return error.InvalidBlockSize;
    if (min_grid_size == 0) return error.InvalidGridSize;
    if (max_blocks == 0) return error.InvalidOccupancy;
}
