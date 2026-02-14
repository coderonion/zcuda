/// zCUDA Unit Tests: types
const std = @import("std");
const cuda = @import("zcuda");
const types = cuda.types;

test "cudaTypeName" {
    try std.testing.expectEqualStrings("float", types.cudaTypeName(f32));
    try std.testing.expectEqualStrings("double", types.cudaTypeName(f64));
    try std.testing.expectEqualStrings("int", types.cudaTypeName(i32));
    try std.testing.expectEqualStrings("unsigned int", types.cudaTypeName(u32));
    try std.testing.expectEqualStrings("size_t", types.cudaTypeName(usize));
    try std.testing.expectEqualStrings("bool", types.cudaTypeName(bool));
}

test "Dim3" {
    const d = types.Dim3.init(2, 3, 4);
    try std.testing.expectEqual(@as(u32, 2), d.x);
    try std.testing.expectEqual(@as(u32, 3), d.y);
    try std.testing.expectEqual(@as(u32, 4), d.z);

    const l = types.Dim3.linear(128);
    try std.testing.expectEqual(@as(u32, 128), l.x);
    try std.testing.expectEqual(@as(u32, 1), l.y);
}

test "LaunchConfig.forNumElems" {
    const cfg = types.LaunchConfig.forNumElems(1000);
    try std.testing.expectEqual(@as(u32, 4), cfg.grid_dim.x);
    try std.testing.expectEqual(@as(u32, 256), cfg.block_dim.x);
    try std.testing.expectEqual(@as(u32, 0), cfg.shared_mem_bytes);
}

test "DevicePtr" {
    const p = types.DevicePtr(f32).init(0x1000);
    try std.testing.expectEqual(@as(usize, 0x1000), p.ptr);
    try std.testing.expect(!p.isNull());

    const p2 = p.offset(10);
    try std.testing.expectEqual(@as(usize, 0x1000 + 10 * 4), p2.ptr);

    const null_ptr = types.DevicePtr(f32).init(0);
    try std.testing.expect(null_ptr.isNull());
}
