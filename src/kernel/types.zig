// src/kernel/types.zig — Device-side type abstractions for GPU kernels
//
// Provides ergonomic, type-safe wrappers for common GPU programming patterns:
//   - DeviceSlice(T): bounds-aware pointer + length pair
//   - DevicePtr(T): single-element device pointer wrapper
//   - GridStrideIterator: grid-stride loop pattern for processing arrays larger than grid size
//   - globalThreadIdx(): convenience 1D global thread index
//
// Usage:
//   const cuda = @import("zcuda_kernel");
//   const types = cuda.types;
//
//   export fn kernel(data: types.DeviceSlice(f32)) callconv(.kernel) void {
//       for (types.gridStrideRange(data.len)) |i| {
//           data.ptr[i] = ...;
//       }
//   }

const intrinsics = @import("intrinsics.zig");

// ============================================================================
// Global Thread Index — convenience 1D index computation
// ============================================================================

/// Compute the global 1D thread index: blockIdx.x * blockDim.x + threadIdx.x
pub inline fn globalThreadIdx() u32 {
    return intrinsics.blockIdx().x * intrinsics.blockDim().x + intrinsics.threadIdx().x;
}

/// Compute the total number of threads in the grid (1D)
pub inline fn gridStride() u32 {
    return intrinsics.blockDim().x * intrinsics.gridDim().x;
}

// ============================================================================
// DeviceSlice — typed pointer + length for safe device memory access
// ============================================================================

/// A device-side slice: a typed pointer with a known length.
/// Mirrors the concept of Zig slices but for device global memory.
pub fn DeviceSlice(comptime T: type) type {
    return extern struct {
        ptr: [*]T,
        len: u32,

        const Self = @This();

        /// Get element at index (no bounds check in release mode)
        pub inline fn get(self: Self, idx: u32) T {
            return self.ptr[idx];
        }

        /// Set element at index
        pub inline fn set(self: Self, idx: u32, val: T) void {
            self.ptr[idx] = val;
        }

        /// Create from raw pointer and length
        pub inline fn init(ptr: [*]T, len: u32) Self {
            return .{ .ptr = ptr, .len = len };
        }
    };
}

// ============================================================================
// DevicePtr — single-value device pointer wrapper
// ============================================================================

/// A single-value device pointer, useful for output scalars (e.g., reduction results).
pub fn DevicePtr(comptime T: type) type {
    return extern struct {
        ptr: *T,

        const Self = @This();

        pub inline fn load(self: Self) T {
            return self.ptr.*;
        }

        pub inline fn store(self: Self, val: T) void {
            self.ptr.* = val;
        }

        /// Atomic add (only for supported types: f32, u32, i32)
        pub inline fn atomicAdd(self: Self, val: T) T {
            return intrinsics.atomicAdd(self.ptr, val);
        }
    };
}

// ============================================================================
// Grid-Stride Loop — common GPU pattern for processing large arrays
// ============================================================================

/// Grid-stride loop iterator for processing arrays larger than the grid size.
///
/// Instead of:
///   const i = blockIdx.x * blockDim.x + threadIdx.x;
///   if (i < n) { ... }
///
/// Use:
///   var iter = cuda.types.gridStrideLoop(n);
///   while (iter.next()) |i| {
///       output[i] = process(input[i]);
///   }
///
/// This automatically handles arrays of any size with a single kernel launch.
pub const GridStrideIterator = struct {
    current: u32,
    stride: u32,
    end: u32,

    /// Advance to the next index, or return null if done.
    pub inline fn next(self: *GridStrideIterator) ?u32 {
        const idx = self.current;
        if (idx >= self.end) return null;
        self.current = idx + self.stride;
        return idx;
    }

    /// Reset the iterator to its initial state
    pub inline fn reset(self: *GridStrideIterator) void {
        self.current = globalThreadIdx();
    }
};

/// Create a grid-stride loop iterator for processing `n` elements.
/// Each thread in the grid will process elements at globalThreadIdx, globalThreadIdx + gridStride, etc.
pub inline fn gridStrideLoop(n: u32) GridStrideIterator {
    return .{
        .current = globalThreadIdx(),
        .stride = gridStride(),
        .end = n,
    };
}
