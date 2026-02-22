// src/kernel/shared_mem.zig — Shared Memory Utilities
//
// CUDA shared memory lives in addrspace(3) on NVPTX. This module provides
// ergonomic wrappers for static (comptime-sized) and extern (dynamic) shared memory.
//
// Usage:
//   const cuda = @import("zcuda_kernel");
//   const smem = cuda.shared_mem;
//
//   // Static shared memory (size known at compile time):
//   const tile = smem.SharedArray(f32, 256);        // 256-element shared array
//   c tile.ptr()[threadIdx] = value;
//   cuda.__syncthreads();
//   const x = tile.ptr()[threadIdx];
//
//   // Dynamic shared memory (size set at launch time):
//   const dyn = smem.dynamicShared(f32);
//   dyn[threadIdx] = value;

const intrinsics = @import("intrinsics.zig");
const builtin = @import("std").builtin;

// ============================================================================
// Static Shared Memory — compile-time sized, placed in addrspace(3)
// ============================================================================

/// A shared memory array of `N` elements of type `T`.
///
/// This leverages Zig's `addrspace(3)` to emit `.shared` section PTX variables.
/// The storage is per-block (not per-thread).
///
/// **IMPORTANT**: Multiple calls with the same `(T, N)` return the same type
/// and share storage. If you need two independent shared arrays of the same
/// type and size, use different sizes or a combined allocation:
/// ```
/// // Option 1: combine into one array and split
/// const tile = SharedArray(f32, TILE * TILE * 2);
/// const sa = tile.ptr();
/// const sb = tile.ptr() + TILE * TILE;
///
/// // Option 2: use different sizes (add padding)
/// const tile_a = SharedArray(f32, TILE * TILE);
/// const tile_b = SharedArray(f32, TILE * TILE + 1); // unique type
/// ```
pub fn SharedArray(comptime T: type, comptime N: u32) type {
    return struct {
        /// The underlying shared memory storage.
        /// `addrspace(3)` translates to `.shared` in PTX on nvptx64.
        var storage: [N]T addrspace(.shared) = undefined;

        /// Get a generic pointer (addrspace-cast) to the shared array.
        /// The returned pointer can be indexed normally.
        pub inline fn ptr() [*]T {
            const arr_ptr: *[N]T = @addrSpaceCast(&storage);
            return @ptrCast(arr_ptr);
        }

        /// Get a slice with bounds info.
        pub inline fn slice() []T {
            const arr_ptr: *[N]T = @addrSpaceCast(&storage);
            return arr_ptr[0..N];
        }

        /// Number of elements.
        pub inline fn len() u32 {
            return N;
        }

        /// Size in bytes.
        pub inline fn sizeBytes() u32 {
            return N * @sizeOf(T);
        }
    };
}

// ============================================================================
// Dynamic Shared Memory — size determined at kernel launch
// ============================================================================

/// Pointer to the base of dynamically-allocated shared memory.
///
/// In CUDA C++, extern __shared__ T arr[];
/// In PTX, this corresponds to a .extern .shared variable.
///
/// The dynamic shared memory size is set via LaunchConfig.shared_mem_bytes
/// on the host side. This function returns a pointer to that region, cast to
/// the requested type.
///
/// Usage:
/// ```
/// const dyn = dynamicShared(f32);
/// dyn[threadIdx().x] = val;
/// ```
pub inline fn dynamicShared(comptime T: type) [*]T {
    return @ptrCast(@alignCast(dynamicSharedBytes()));
}

/// Dynamic shared memory with offset (for multiple arrays in the same dynamic region).
///
/// When you need multiple dynamic shared arrays, partition them manually:
/// ```
/// const base = dynamicSharedBytes();
/// const arr_a: [*]f32 = @ptrCast(@alignCast(base));
/// const arr_b: [*]f32 = @ptrCast(@alignCast(base + 1024));
/// ```
pub inline fn dynamicSharedBytes() [*]u8 {
    // Use inline asm with module-scope .extern declaration via a separate
    // module_asm directive. The extern shared memory symbol __dynamic_smem
    // must be declared at module scope in PTX, not inside a function.
    // We reference it via mov.u64 and cast address space 3 (shared) → generic.
    const ptr = asm volatile (
        \\mov.u64 %[ret], __dynamic_smem;
        : [ret] "=l" (-> [*]u8),
    );
    return ptr;
}

// Module-scope declaration: generate .extern .shared at PTX module level.
// This comptime block emits the declaration outside any function body.
comptime {
    asm (".extern .shared .align 16 .b8 __dynamic_smem[];");
}

// ============================================================================
// Shared Memory Utilities
// ============================================================================

/// Zero-fill a static shared array cooperatively (all threads in block participate).
/// Must be followed by __syncthreads() before reading.
pub inline fn clearShared(comptime T: type, sptr: [*]T, num_elements: u32) void {
    const tid = intrinsics.threadIdx().x;
    const block_size = intrinsics.blockDim().x;
    var i = tid;
    while (i < num_elements) : (i += block_size) {
        sptr[i] = @as(T, 0);
    }
}

/// Cooperative load from global to shared memory.
/// All threads in the block participate. Must __syncthreads() after.
pub inline fn loadToShared(comptime T: type, dst: [*]T, src: [*]const T, num_elements: u32) void {
    const tid = intrinsics.threadIdx().x;
    const block_size = intrinsics.blockDim().x;
    var i = tid;
    while (i < num_elements) : (i += block_size) {
        dst[i] = src[i];
    }
}

/// Cooperative store from shared to global memory.
/// All threads in the block participate.
pub inline fn storeFromShared(comptime T: type, dst: [*]T, src: [*]const T, num_elements: u32) void {
    const tid = intrinsics.threadIdx().x;
    const block_size = intrinsics.blockDim().x;
    var i = tid;
    while (i < num_elements) : (i += block_size) {
        dst[i] = src[i];
    }
}

/// Warp-level shared memory reduction (sum).
/// Reduces `n` elements in shared memory using tree reduction.
/// Assumes n is a power of 2 and n <= blockDim.x.
pub inline fn reduceSum(comptime T: type, sdata: [*]T, tid: u32, n: u32) void {
    var stride = n >> 1;
    while (stride > 0) : (stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        intrinsics.__syncthreads();
    }
}
