/// zCUDA Unit Tests: Kernel shared types (Vec2/3/4, Matrix, LaunchConfig)
///
/// These types are pure Zig — no GPU or inline asm — so they can be
/// tested on the host without CUDA hardware.
const std = @import("std");
const shared = @import("shared_types");

// ============================================================================
// Vec2 Tests
// ============================================================================

test "Vec2.init" {
    const v = shared.Vec2.init(3.0, 4.0);
    try std.testing.expectEqual(@as(f32, 3.0), v.x);
    try std.testing.expectEqual(@as(f32, 4.0), v.y);
}

test "Vec2.add" {
    const a = shared.Vec2.init(1.0, 2.0);
    const b = shared.Vec2.init(3.0, 4.0);
    const c = shared.Vec2.add(a, b);
    try std.testing.expectEqual(@as(f32, 4.0), c.x);
    try std.testing.expectEqual(@as(f32, 6.0), c.y);
}

test "Vec2.scale" {
    const v = shared.Vec2.init(2.0, 3.0);
    const s = shared.Vec2.scale(v, 2.0);
    try std.testing.expectEqual(@as(f32, 4.0), s.x);
    try std.testing.expectEqual(@as(f32, 6.0), s.y);
}

test "Vec2.dot" {
    const a = shared.Vec2.init(1.0, 2.0);
    const b = shared.Vec2.init(3.0, 4.0);
    try std.testing.expectEqual(@as(f32, 11.0), shared.Vec2.dot(a, b));
}

test "Vec2.dot orthogonal" {
    const a = shared.Vec2.init(1.0, 0.0);
    const b = shared.Vec2.init(0.0, 1.0);
    try std.testing.expectEqual(@as(f32, 0.0), shared.Vec2.dot(a, b));
}

// ============================================================================
// Vec3 Tests
// ============================================================================

test "Vec3.init" {
    const v = shared.Vec3.init(1.0, 2.0, 3.0);
    try std.testing.expectEqual(@as(f32, 1.0), v.x);
    try std.testing.expectEqual(@as(f32, 2.0), v.y);
    try std.testing.expectEqual(@as(f32, 3.0), v.z);
}

test "Vec3.add" {
    const a = shared.Vec3.init(1.0, 2.0, 3.0);
    const b = shared.Vec3.init(4.0, 5.0, 6.0);
    const c = shared.Vec3.add(a, b);
    try std.testing.expectEqual(@as(f32, 5.0), c.x);
    try std.testing.expectEqual(@as(f32, 7.0), c.y);
    try std.testing.expectEqual(@as(f32, 9.0), c.z);
}

test "Vec3.sub" {
    const a = shared.Vec3.init(4.0, 5.0, 6.0);
    const b = shared.Vec3.init(1.0, 2.0, 3.0);
    const c = shared.Vec3.sub(a, b);
    try std.testing.expectEqual(@as(f32, 3.0), c.x);
    try std.testing.expectEqual(@as(f32, 3.0), c.y);
    try std.testing.expectEqual(@as(f32, 3.0), c.z);
}

test "Vec3.scale" {
    const v = shared.Vec3.init(1.0, 2.0, 3.0);
    const s = shared.Vec3.scale(v, -1.0);
    try std.testing.expectEqual(@as(f32, -1.0), s.x);
    try std.testing.expectEqual(@as(f32, -2.0), s.y);
    try std.testing.expectEqual(@as(f32, -3.0), s.z);
}

test "Vec3.dot" {
    const a = shared.Vec3.init(1.0, 2.0, 3.0);
    const b = shared.Vec3.init(4.0, 5.0, 6.0);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try std.testing.expectEqual(@as(f32, 32.0), shared.Vec3.dot(a, b));
}

test "Vec3.dot self = length²" {
    const v = shared.Vec3.init(3.0, 4.0, 0.0);
    try std.testing.expectEqual(@as(f32, 25.0), shared.Vec3.dot(v, v));
}

test "Vec3.cross basis vectors" {
    const i = shared.Vec3.init(1.0, 0.0, 0.0);
    const j = shared.Vec3.init(0.0, 1.0, 0.0);
    const k = shared.Vec3.cross(i, j);
    // i × j = k
    try std.testing.expectEqual(@as(f32, 0.0), k.x);
    try std.testing.expectEqual(@as(f32, 0.0), k.y);
    try std.testing.expectEqual(@as(f32, 1.0), k.z);
}

test "Vec3.cross anticommutative" {
    const a = shared.Vec3.init(1.0, 2.0, 3.0);
    const b = shared.Vec3.init(4.0, 5.0, 6.0);
    const ab = shared.Vec3.cross(a, b);
    const ba = shared.Vec3.cross(b, a);
    // a × b = -(b × a)
    try std.testing.expectEqual(ab.x, -ba.x);
    try std.testing.expectEqual(ab.y, -ba.y);
    try std.testing.expectEqual(ab.z, -ba.z);
}

test "Vec3.cross self = zero" {
    const v = shared.Vec3.init(1.0, 2.0, 3.0);
    const c = shared.Vec3.cross(v, v);
    try std.testing.expectEqual(@as(f32, 0.0), c.x);
    try std.testing.expectEqual(@as(f32, 0.0), c.y);
    try std.testing.expectEqual(@as(f32, 0.0), c.z);
}

// ============================================================================
// Vec4 Tests
// ============================================================================

test "Vec4.init" {
    const v = shared.Vec4.init(1.0, 2.0, 3.0, 4.0);
    try std.testing.expectEqual(@as(f32, 1.0), v.x);
    try std.testing.expectEqual(@as(f32, 4.0), v.w);
}

test "Vec4.add" {
    const a = shared.Vec4.init(1.0, 2.0, 3.0, 4.0);
    const b = shared.Vec4.init(5.0, 6.0, 7.0, 8.0);
    const c = shared.Vec4.add(a, b);
    try std.testing.expectEqual(@as(f32, 6.0), c.x);
    try std.testing.expectEqual(@as(f32, 8.0), c.y);
    try std.testing.expectEqual(@as(f32, 10.0), c.z);
    try std.testing.expectEqual(@as(f32, 12.0), c.w);
}

test "Vec4.dot" {
    const a = shared.Vec4.init(1.0, 2.0, 3.0, 4.0);
    const b = shared.Vec4.init(5.0, 6.0, 7.0, 8.0);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    try std.testing.expectEqual(@as(f32, 70.0), shared.Vec4.dot(a, b));
}

// ============================================================================
// Int2/Int3 Tests
// ============================================================================

test "Int2.init" {
    const v = shared.Int2.init(10, 20);
    try std.testing.expectEqual(@as(i32, 10), v.x);
    try std.testing.expectEqual(@as(i32, 20), v.y);
}

test "Int3.init" {
    const v = shared.Int3.init(128, 256, 1);
    try std.testing.expectEqual(@as(i32, 128), v.x);
    try std.testing.expectEqual(@as(i32, 256), v.y);
    try std.testing.expectEqual(@as(i32, 1), v.z);
}

// ============================================================================
// Matrix Tests
// ============================================================================

test "Matrix3x3.identity" {
    const m = shared.Matrix3x3.identity();
    // Diagonal = 1
    try std.testing.expectEqual(@as(f32, 1.0), m.get(0, 0));
    try std.testing.expectEqual(@as(f32, 1.0), m.get(1, 1));
    try std.testing.expectEqual(@as(f32, 1.0), m.get(2, 2));
    // Off-diagonal = 0
    try std.testing.expectEqual(@as(f32, 0.0), m.get(0, 1));
    try std.testing.expectEqual(@as(f32, 0.0), m.get(1, 0));
    try std.testing.expectEqual(@as(f32, 0.0), m.get(2, 0));
}

test "Matrix3x3.set and get" {
    var m = shared.Matrix3x3.identity();
    m.set(0, 2, 42.0);
    try std.testing.expectEqual(@as(f32, 42.0), m.get(0, 2));
    // Other elements unchanged
    try std.testing.expectEqual(@as(f32, 1.0), m.get(0, 0));
}

test "Matrix4x4.identity" {
    const m = shared.Matrix4x4.identity();
    try std.testing.expectEqual(@as(f32, 1.0), m.get(0, 0));
    try std.testing.expectEqual(@as(f32, 1.0), m.get(3, 3));
    try std.testing.expectEqual(@as(f32, 0.0), m.get(0, 3));
    try std.testing.expectEqual(@as(f32, 0.0), m.get(3, 0));
}

test "Matrix4x4.set and get" {
    var m = shared.Matrix4x4.identity();
    m.set(1, 3, -5.0);
    try std.testing.expectEqual(@as(f32, -5.0), m.get(1, 3));
}

// ============================================================================
// LaunchConfig Tests
// ============================================================================

test "LaunchConfig.init1D" {
    const cfg = shared.LaunchConfig.init1D(4, 256);
    try std.testing.expectEqual(@as(u32, 4), cfg.grid_dim_x);
    try std.testing.expectEqual(@as(u32, 256), cfg.block_dim_x);
    try std.testing.expectEqual(@as(u32, 1), cfg.grid_dim_y);
    try std.testing.expectEqual(@as(u32, 1), cfg.block_dim_y);
}

test "LaunchConfig.init2D" {
    const cfg = shared.LaunchConfig.init2D(8, 8, 16, 16);
    try std.testing.expectEqual(@as(u32, 8), cfg.grid_dim_x);
    try std.testing.expectEqual(@as(u32, 8), cfg.grid_dim_y);
    try std.testing.expectEqual(@as(u32, 16), cfg.block_dim_x);
    try std.testing.expectEqual(@as(u32, 16), cfg.block_dim_y);
}

test "LaunchConfig.forElementCount" {
    // 1000 elements / 256 block = ceil(3.9) = 4 blocks
    const cfg = shared.LaunchConfig.forElementCount(1000, 256);
    try std.testing.expectEqual(@as(u32, 4), cfg.grid_dim_x);
    try std.testing.expectEqual(@as(u32, 256), cfg.block_dim_x);
}

test "LaunchConfig.forElementCount exact" {
    // 512 elements / 256 block = exactly 2 blocks
    const cfg = shared.LaunchConfig.forElementCount(512, 256);
    try std.testing.expectEqual(@as(u32, 2), cfg.grid_dim_x);
}

test "LaunchConfig.forElementCount single element" {
    const cfg = shared.LaunchConfig.forElementCount(1, 256);
    try std.testing.expectEqual(@as(u32, 1), cfg.grid_dim_x);
}

test "LaunchConfig default shared_mem_bytes is zero" {
    const cfg = shared.LaunchConfig.init1D(1, 128);
    try std.testing.expectEqual(@as(u32, 0), cfg.shared_mem_bytes);
}

// ============================================================================
// Struct Layout Tests — verify extern struct compatibility
// ============================================================================

test "Vec2 size is 8 bytes (2 × f32)" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(shared.Vec2));
}

test "Vec3 size is 12 bytes (3 × f32)" {
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(shared.Vec3));
}

test "Vec4 size is 16 bytes (4 × f32)" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(shared.Vec4));
}

test "Matrix3x3 size is 36 bytes (9 × f32)" {
    try std.testing.expectEqual(@as(usize, 36), @sizeOf(shared.Matrix3x3));
}

test "Matrix4x4 size is 64 bytes (16 × f32)" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(shared.Matrix4x4));
}
