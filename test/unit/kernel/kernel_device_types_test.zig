/// zCUDA Unit Tests: Device types (DeviceSlice, DevicePtr, GridStrideIterator)
///
/// Tests comptime type properties AND runtime correctness using host memory.
/// DeviceSlice.get/set/init and DevicePtr.load/store are pure pointer ops —
/// fully testable on host without GPU.
const std = @import("std");
const types = @import("types");

// ============================================================================
// DeviceSlice — comptime type properties
// ============================================================================

test "DeviceSlice(f32) is an extern struct with ptr and len fields" {
    const SliceF32 = types.DeviceSlice(f32);
    const info = @typeInfo(SliceF32);
    try std.testing.expect(info == .@"struct");
    const fields = info.@"struct".fields;
    try std.testing.expectEqual(@as(usize, 2), fields.len);
    try std.testing.expectEqualStrings("ptr", fields[0].name);
    try std.testing.expectEqualStrings("len", fields[1].name);
}

test "DeviceSlice(f32) layout is extern" {
    const info = @typeInfo(types.DeviceSlice(f32));
    try std.testing.expectEqual(std.builtin.Type.ContainerLayout.@"extern", info.@"struct".layout);
}

test "DeviceSlice generic instantiation produces distinct types" {
    const S1 = types.DeviceSlice(f32);
    const S2 = types.DeviceSlice(u32);
    const S3 = types.DeviceSlice(i64);
    try std.testing.expect(S1 != S2);
    try std.testing.expect(S2 != S3);
    try std.testing.expect(S1 != S3);
}

test "DeviceSlice(f32) has get, set, init methods" {
    const SliceF32 = types.DeviceSlice(f32);
    try std.testing.expect(@hasDecl(SliceF32, "get"));
    try std.testing.expect(@hasDecl(SliceF32, "set"));
    try std.testing.expect(@hasDecl(SliceF32, "init"));
}

// ============================================================================
// DeviceSlice — runtime correctness on host memory
// ============================================================================

test "DeviceSlice.init sets ptr and len" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const slice = types.DeviceSlice(f32).init(&data, 4);
    try std.testing.expectEqual(@as(u32, 4), slice.len);
    try std.testing.expectEqual(@as(*f32, &data[0]), @as(*f32, @ptrCast(slice.ptr)));
}

test "DeviceSlice.get reads correct values" {
    var data = [_]f32{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    const slice = types.DeviceSlice(f32).init(&data, 5);
    try std.testing.expectEqual(@as(f32, 10.0), slice.get(0));
    try std.testing.expectEqual(@as(f32, 20.0), slice.get(1));
    try std.testing.expectEqual(@as(f32, 30.0), slice.get(2));
    try std.testing.expectEqual(@as(f32, 40.0), slice.get(3));
    try std.testing.expectEqual(@as(f32, 50.0), slice.get(4));
}

test "DeviceSlice.set writes correct values" {
    var data = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const slice = types.DeviceSlice(f32).init(&data, 4);
    slice.set(0, 100.0);
    slice.set(1, 200.0);
    slice.set(2, 300.0);
    slice.set(3, 400.0);
    try std.testing.expectEqual(@as(f32, 100.0), data[0]);
    try std.testing.expectEqual(@as(f32, 200.0), data[1]);
    try std.testing.expectEqual(@as(f32, 300.0), data[2]);
    try std.testing.expectEqual(@as(f32, 400.0), data[3]);
}

test "DeviceSlice get/set round-trip" {
    var data = [_]f32{0.0} ** 8;
    const slice = types.DeviceSlice(f32).init(&data, 8);
    // Write pattern
    for (0..8) |i| {
        slice.set(@intCast(i), @as(f32, @floatFromInt(i)) * 3.14);
    }
    // Read back and verify
    for (0..8) |i| {
        const expected = @as(f32, @floatFromInt(i)) * 3.14;
        try std.testing.expectEqual(expected, slice.get(@intCast(i)));
    }
}

test "DeviceSlice(u32) get/set correctness" {
    var data = [_]u32{ 0, 0, 0, 0, 0 };
    const slice = types.DeviceSlice(u32).init(&data, 5);
    slice.set(0, 0xDEADBEEF);
    slice.set(4, 0xCAFEBABE);
    try std.testing.expectEqual(@as(u32, 0xDEADBEEF), slice.get(0));
    try std.testing.expectEqual(@as(u32, 0), slice.get(1));
    try std.testing.expectEqual(@as(u32, 0xCAFEBABE), slice.get(4));
}

// ============================================================================
// DevicePtr — comptime type properties
// ============================================================================

test "DevicePtr(f32) is an extern struct with ptr field" {
    const PtrF32 = types.DevicePtr(f32);
    const info = @typeInfo(PtrF32);
    try std.testing.expect(info == .@"struct");
    try std.testing.expectEqual(std.builtin.Type.ContainerLayout.@"extern", info.@"struct".layout);
    const fields = info.@"struct".fields;
    try std.testing.expectEqual(@as(usize, 1), fields.len);
    try std.testing.expectEqualStrings("ptr", fields[0].name);
}

test "DevicePtr(f32) has load, store, atomicAdd methods" {
    const PtrF32 = types.DevicePtr(f32);
    try std.testing.expect(@hasDecl(PtrF32, "load"));
    try std.testing.expect(@hasDecl(PtrF32, "store"));
    try std.testing.expect(@hasDecl(PtrF32, "atomicAdd"));
}

// ============================================================================
// DevicePtr — runtime correctness on host memory
// ============================================================================

test "DevicePtr.load reads correct value" {
    var val: f32 = 42.5;
    const ptr = types.DevicePtr(f32){ .ptr = &val };
    try std.testing.expectEqual(@as(f32, 42.5), ptr.load());
}

test "DevicePtr.store writes correct value" {
    var val: f32 = 0.0;
    const ptr = types.DevicePtr(f32){ .ptr = &val };
    ptr.store(99.9);
    try std.testing.expectEqual(@as(f32, 99.9), val);
}

test "DevicePtr load/store round-trip" {
    var val: f32 = 1.23;
    const ptr = types.DevicePtr(f32){ .ptr = &val };
    try std.testing.expectEqual(@as(f32, 1.23), ptr.load());
    ptr.store(4.56);
    try std.testing.expectEqual(@as(f32, 4.56), ptr.load());
    ptr.store(-7.89);
    try std.testing.expectEqual(@as(f32, -7.89), ptr.load());
}

test "DevicePtr(u32) load/store correctness" {
    var val: u32 = 0;
    const ptr = types.DevicePtr(u32){ .ptr = &val };
    ptr.store(0xFFFF_FFFF);
    try std.testing.expectEqual(@as(u32, 0xFFFF_FFFF), ptr.load());
    ptr.store(0);
    try std.testing.expectEqual(@as(u32, 0), ptr.load());
}

test "DevicePtr(i32) load/store negative values" {
    var val: i32 = 0;
    const ptr = types.DevicePtr(i32){ .ptr = &val };
    ptr.store(-12345);
    try std.testing.expectEqual(@as(i32, -12345), ptr.load());
}

// ============================================================================
// GridStrideIterator — type info
// ============================================================================

test "GridStrideIterator struct has current, stride, end fields" {
    const info = @typeInfo(types.GridStrideIterator);
    const fields = info.@"struct".fields;
    try std.testing.expectEqual(@as(usize, 3), fields.len);
    try std.testing.expectEqualStrings("current", fields[0].name);
    try std.testing.expectEqualStrings("stride", fields[1].name);
    try std.testing.expectEqualStrings("end", fields[2].name);
}

test "GridStrideIterator has next and reset methods" {
    try std.testing.expect(@hasDecl(types.GridStrideIterator, "next"));
    try std.testing.expect(@hasDecl(types.GridStrideIterator, "reset"));
}
