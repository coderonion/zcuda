// src/kernel/shared_types.zig — Host-device shared type definitions
//
// These types use `extern struct` to guarantee C-compatible memory layout,
// making them safe to pass between host (CPU) and device (GPU) code.
//
// Both host and device code should import this same module to ensure
// layout consistency.
//
// Usage:
//   // In kernel code:
//   const cuda = @import("zcuda_kernel");
//   const Vec3 = cuda.shared.Vec3;
//
//   // In host code:
//   const shared = @import("zcuda").shared_types;
//   const Vec3 = shared.Vec3;

// ============================================================================
// Vector Types — matching CUDA's float2/float3/float4
// ============================================================================

/// 2D float vector, compatible with CUDA float2
pub const Vec2 = extern struct {
    x: f32 = 0,
    y: f32 = 0,

    pub inline fn init(x: f32, y: f32) Vec2 {
        return .{ .x = x, .y = y };
    }

    pub inline fn add(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x + b.x, .y = a.y + b.y };
    }

    pub inline fn scale(v: Vec2, s: f32) Vec2 {
        return .{ .x = v.x * s, .y = v.y * s };
    }

    pub inline fn dot(a: Vec2, b: Vec2) f32 {
        return a.x * b.x + a.y * b.y;
    }
};

/// 3D float vector, compatible with CUDA float3
pub const Vec3 = extern struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,

    pub inline fn init(x: f32, y: f32, z: f32) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub inline fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }

    pub inline fn sub(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
    }

    pub inline fn scale(v: Vec3, s: f32) Vec3 {
        return .{ .x = v.x * s, .y = v.y * s, .z = v.z * s };
    }

    pub inline fn dot(a: Vec3, b: Vec3) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    pub inline fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }
};

/// 4D float vector, compatible with CUDA float4
pub const Vec4 = extern struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 0,

    pub inline fn init(x: f32, y: f32, z: f32, w: f32) Vec4 {
        return .{ .x = x, .y = y, .z = z, .w = w };
    }

    pub inline fn add(a: Vec4, b: Vec4) Vec4 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z, .w = a.w + b.w };
    }

    pub inline fn dot(a: Vec4, b: Vec4) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
};

// ============================================================================
// Integer Vector Types
// ============================================================================

/// 2D integer vector, compatible with CUDA int2
pub const Int2 = extern struct {
    x: i32 = 0,
    y: i32 = 0,

    pub inline fn init(x: i32, y: i32) Int2 {
        return .{ .x = x, .y = y };
    }
};

/// 3D integer vector, compatible with CUDA int3/dim3
pub const Int3 = extern struct {
    x: i32 = 0,
    y: i32 = 0,
    z: i32 = 0,

    pub inline fn init(x: i32, y: i32, z: i32) Int3 {
        return .{ .x = x, .y = y, .z = z };
    }
};

// ============================================================================
// Matrix Types — small fixed-size matrices for GPU computation
// ============================================================================

/// 3×3 float matrix (row-major), common in graphics and physics
pub const Matrix3x3 = extern struct {
    data: [9]f32,

    pub inline fn identity() Matrix3x3 {
        return .{ .data = .{
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        } };
    }

    pub inline fn get(self: Matrix3x3, row: u32, col: u32) f32 {
        return self.data[row * 3 + col];
    }

    pub inline fn set(self: *Matrix3x3, row: u32, col: u32, val: f32) void {
        self.data[row * 3 + col] = val;
    }
};

/// 4×4 float matrix (row-major), common in graphics transformations
pub const Matrix4x4 = extern struct {
    data: [16]f32,

    pub inline fn identity() Matrix4x4 {
        return .{ .data = .{
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        } };
    }

    pub inline fn get(self: Matrix4x4, row: u32, col: u32) f32 {
        return self.data[row * 4 + col];
    }

    pub inline fn set(self: *Matrix4x4, row: u32, col: u32, val: f32) void {
        self.data[row * 4 + col] = val;
    }
};

// ============================================================================
// Launch Configuration
// ============================================================================

/// Kernel launch configuration — matches cudaLaunchKernel parameters
pub const LaunchConfig = extern struct {
    grid_dim_x: u32 = 1,
    grid_dim_y: u32 = 1,
    grid_dim_z: u32 = 1,
    block_dim_x: u32 = 256,
    block_dim_y: u32 = 1,
    block_dim_z: u32 = 1,
    shared_mem_bytes: u32 = 0,

    /// Create a 1D launch config: (gridSize, blockSize)
    pub inline fn init1D(grid_size: u32, block_size: u32) LaunchConfig {
        return .{ .grid_dim_x = grid_size, .block_dim_x = block_size };
    }

    /// Create a 2D launch config
    pub inline fn init2D(gx: u32, gy: u32, bx: u32, by: u32) LaunchConfig {
        return .{ .grid_dim_x = gx, .grid_dim_y = gy, .block_dim_x = bx, .block_dim_y = by };
    }

    /// Calculate optimal 1D launch config for processing `n` elements
    pub inline fn forElementCount(n: u32, block_size: u32) LaunchConfig {
        const grid_size = (n + block_size - 1) / block_size;
        return init1D(grid_size, block_size);
    }
};
