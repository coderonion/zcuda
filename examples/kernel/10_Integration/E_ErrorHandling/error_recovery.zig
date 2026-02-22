// examples/kernel/10_Integration/E_ErrorHandling/error_recovery.zig
// Reference: cuda-samples/0_Introduction/simpleAssert
// API: driver error codes, try/catch patterns
//
// ── Error recovery demonstrations ──
// Demonstrates handling invalid module data and wrong function names.

const std = @import("std");
const cuda = @import("zcuda");
const driver = cuda.driver;

const kernel_vector_add = @import("kernel_vector_add");

/// Demonstrates error handling patterns with zcuda.
pub fn main() !void {
    var ctx = try driver.CudaContext.new(0);
    defer ctx.deinit();
    var stream = try ctx.newStream();
    defer stream.deinit();

    // Normal allocation — should succeed
    var d_buf = try stream.alloc(f32, std.heap.page_allocator, 1024);
    defer d_buf.deinit();

    // Attempt to load invalid module data — expect error
    const result = ctx.loadModule("this is not valid PTX data");
    if (result) |mod| {
        mod.deinit();
        return error.ShouldHaveFailed;
    } else |_| {
        // Expected: invalid PTX data error — successfully caught
    }

    // Load valid module via bridge, then try getting a function with wrong name
    const module = try kernel_vector_add.load(ctx, std.heap.page_allocator);
    defer module.deinit();
    const bad_func = module.getFunction("doesNotExist");
    if (bad_func) |_| {
        return error.ShouldHaveFailed;
    } else |_| {
        // Expected: function not found error — successfully caught
    }
}
