/// zCUDA: Shared type definitions.
///
/// Provides type mappings between Zig types and CUDA C++ kernel types,
/// as well as common structures used across all zCUDA modules.
const std = @import("std");

// ============================================================================
// CUDA Type Name Mapping
// ============================================================================

/// Returns the CUDA C++ type name string corresponding to a Zig type.
/// For example, `f32` maps to `"float"`, `u32` maps to `"unsigned int"`.
pub fn cudaTypeName(comptime T: type) []const u8 {
    return switch (T) {
        bool => "bool",
        i8 => "char",
        i16 => "short",
        i32 => "int",
        i64 => "long",
        isize => "intptr_t",
        u8 => "unsigned char",
        u16 => "unsigned short",
        u32 => "unsigned int",
        u64 => "unsigned long",
        usize => "size_t",
        f32 => "float",
        f64 => "double",
        else => @compileError("Unsupported CUDA type: " ++ @typeName(T)),
    };
}

// ============================================================================
// Launch Configuration
// ============================================================================

/// Three-dimensional size, used for grid and block dimensions.
pub const Dim3 = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,

    pub fn init(x: u32, y: u32, z: u32) Dim3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn linear(n: u32) Dim3 {
        return .{ .x = n, .y = 1, .z = 1 };
    }
};

/// Kernel launch configuration specifying grid dimensions, block dimensions,
/// and shared memory size.
pub const LaunchConfig = struct {
    grid_dim: Dim3 = .{},
    block_dim: Dim3 = .{},
    shared_mem_bytes: u32 = 0,

    /// Create a launch configuration suitable for processing `num_elems` elements.
    /// Uses 256 threads per block by default.
    pub fn forNumElems(num_elems: u32) LaunchConfig {
        const threads_per_block: u32 = 256;
        const num_blocks = (num_elems + threads_per_block - 1) / threads_per_block;
        return .{
            .grid_dim = Dim3.linear(num_blocks),
            .block_dim = Dim3.linear(threads_per_block),
            .shared_mem_bytes = 0,
        };
    }

    /// Create a launch configuration with custom threads per block.
    pub fn forNumElemsCustom(num_elems: u32, threads_per_block: u32) LaunchConfig {
        const num_blocks = (num_elems + threads_per_block - 1) / threads_per_block;
        return .{
            .grid_dim = Dim3.linear(num_blocks),
            .block_dim = Dim3.linear(threads_per_block),
            .shared_mem_bytes = 0,
        };
    }
};

// ============================================================================
// Device Pointer
// ============================================================================

/// A typed device pointer representing a location in GPU memory.
pub fn DevicePtr(comptime T: type) type {
    return struct {
        ptr: usize,

        const Self = @This();

        pub fn init(ptr: usize) Self {
            return .{ .ptr = ptr };
        }

        pub fn isNull(self: Self) bool {
            return self.ptr == 0;
        }

        /// Offset the pointer by `n` elements of type T.
        pub fn offset(self: Self, n: usize) Self {
            return .{ .ptr = self.ptr + n * @sizeOf(T) };
        }
    };
}

// ============================================================================
// cuBLAS Types
// ============================================================================

/// cuBLAS matrix transpose operation.
pub const Operation = enum {
    no_transpose,
    transpose,
    conj_transpose,
};

/// cuBLAS matrix fill mode (upper or lower triangular).
pub const FillMode = enum {
    lower,
    upper,
};

/// cuBLAS diagonal type.
pub const DiagType = enum {
    non_unit,
    unit,
};

/// cuBLAS side mode (left or right multiply).
pub const SideMode = enum {
    left,
    right,
};

// ============================================================================
// Tests
// ============================================================================
