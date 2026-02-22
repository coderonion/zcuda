/// zCUDA Unit Tests: shared_mem.zig comptime properties
///
/// Tests SharedArray comptime-evaluable properties: len, sizeBytes,
/// and verifies the generic type construction works for different types.
/// Runtime functions (ptr, slice, dynamicShared) require GPU addrspace(3).
const std = @import("std");
const shared_mem = @import("shared_mem");

// ============================================================================
// SharedArray — comptime properties
// ============================================================================

test "SharedArray(f32, 256).len() == 256" {
    const Tile = shared_mem.SharedArray(f32, 256);
    try std.testing.expectEqual(@as(u32, 256), Tile.len());
}

test "SharedArray(f32, 256).sizeBytes() == 1024" {
    const Tile = shared_mem.SharedArray(f32, 256);
    // 256 elements × 4 bytes/element = 1024
    try std.testing.expectEqual(@as(u32, 1024), Tile.sizeBytes());
}

test "SharedArray(u8, 1024).len() == 1024" {
    const Tile = shared_mem.SharedArray(u8, 1024);
    try std.testing.expectEqual(@as(u32, 1024), Tile.len());
}

test "SharedArray(u8, 1024).sizeBytes() == 1024" {
    const Tile = shared_mem.SharedArray(u8, 1024);
    // 1024 elements × 1 byte/element = 1024
    try std.testing.expectEqual(@as(u32, 1024), Tile.sizeBytes());
}

test "SharedArray(f64, 128).sizeBytes() == 1024" {
    const Tile = shared_mem.SharedArray(f64, 128);
    // 128 elements × 8 bytes/element = 1024
    try std.testing.expectEqual(@as(u32, 1024), Tile.sizeBytes());
}

test "SharedArray(u32, 512).len() == 512" {
    const Tile = shared_mem.SharedArray(u32, 512);
    try std.testing.expectEqual(@as(u32, 512), Tile.len());
}

test "SharedArray(u32, 512).sizeBytes() == 2048" {
    const Tile = shared_mem.SharedArray(u32, 512);
    // 512 elements × 4 bytes/element = 2048
    try std.testing.expectEqual(@as(u32, 2048), Tile.sizeBytes());
}

// ============================================================================
// SharedArray — method existence
// ============================================================================

test "SharedArray(f32, 256) has ptr, slice, len, sizeBytes methods" {
    const Tile = shared_mem.SharedArray(f32, 256);
    try std.testing.expect(@hasDecl(Tile, "ptr"));
    try std.testing.expect(@hasDecl(Tile, "slice"));
    try std.testing.expect(@hasDecl(Tile, "len"));
    try std.testing.expect(@hasDecl(Tile, "sizeBytes"));
}

// ============================================================================
// Module-level function existence
// ============================================================================

test "shared_mem exports dynamicShared" {
    try std.testing.expect(@hasDecl(shared_mem, "dynamicShared"));
}

test "shared_mem exports dynamicSharedBytes" {
    try std.testing.expect(@hasDecl(shared_mem, "dynamicSharedBytes"));
}

test "shared_mem exports clearShared" {
    try std.testing.expect(@hasDecl(shared_mem, "clearShared"));
}

test "shared_mem exports loadToShared" {
    try std.testing.expect(@hasDecl(shared_mem, "loadToShared"));
}

test "shared_mem exports storeFromShared" {
    try std.testing.expect(@hasDecl(shared_mem, "storeFromShared"));
}

test "shared_mem exports reduceSum" {
    try std.testing.expect(@hasDecl(shared_mem, "reduceSum"));
}

// ============================================================================
// SharedArray — different N values produce consistent results
// ============================================================================

test "SharedArray sizeBytes consistency across types" {
    // f32 × N should always equal N × 4
    const n_values = [_]u32{ 1, 16, 64, 128, 256, 512, 1024 };
    inline for (n_values) |n| {
        const Tile = shared_mem.SharedArray(f32, n);
        try std.testing.expectEqual(n * 4, Tile.sizeBytes());
        try std.testing.expectEqual(n, Tile.len());
    }
}
