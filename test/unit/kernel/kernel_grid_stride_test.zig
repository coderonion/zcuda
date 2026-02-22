/// zCUDA Unit Tests: GridStrideIterator pure logic
///
/// Tests the state machine in types.GridStrideIterator without GPU.
/// The iterator is pure Zig (no inline asm), fully testable on host.
const std = @import("std");
const types = @import("types");

// ============================================================================
// GridStrideIterator — standard case
// ============================================================================

test "GridStrideIterator standard: start=0, stride=256, end=1000" {
    var iter = types.GridStrideIterator{
        .current = 0,
        .stride = 256,
        .end = 1000,
    };
    try std.testing.expectEqual(@as(?u32, 0), iter.next());
    try std.testing.expectEqual(@as(?u32, 256), iter.next());
    try std.testing.expectEqual(@as(?u32, 512), iter.next());
    try std.testing.expectEqual(@as(?u32, 768), iter.next());
    try std.testing.expectEqual(@as(?u32, null), iter.next());
}

// ============================================================================
// GridStrideIterator — single element
// ============================================================================

test "GridStrideIterator single element: start=0, stride=1, end=1" {
    var iter = types.GridStrideIterator{
        .current = 0,
        .stride = 1,
        .end = 1,
    };
    try std.testing.expectEqual(@as(?u32, 0), iter.next());
    try std.testing.expectEqual(@as(?u32, null), iter.next());
}

// ============================================================================
// GridStrideIterator — empty range
// ============================================================================

test "GridStrideIterator empty range: start=5, stride=1, end=5" {
    var iter = types.GridStrideIterator{
        .current = 5,
        .stride = 1,
        .end = 5,
    };
    try std.testing.expectEqual(@as(?u32, null), iter.next());
}

test "GridStrideIterator empty range: start > end" {
    var iter = types.GridStrideIterator{
        .current = 10,
        .stride = 1,
        .end = 5,
    };
    try std.testing.expectEqual(@as(?u32, null), iter.next());
}

// ============================================================================
// GridStrideIterator — large stride
// ============================================================================

test "GridStrideIterator large stride: stride > end" {
    var iter = types.GridStrideIterator{
        .current = 0,
        .stride = 1024,
        .end = 100,
    };
    try std.testing.expectEqual(@as(?u32, 0), iter.next());
    try std.testing.expectEqual(@as(?u32, null), iter.next());
}

// ============================================================================
// GridStrideIterator — stride exactly divides range
// ============================================================================

test "GridStrideIterator exact division: 0..1024 by 256" {
    var iter = types.GridStrideIterator{
        .current = 0,
        .stride = 256,
        .end = 1024,
    };
    try std.testing.expectEqual(@as(?u32, 0), iter.next());
    try std.testing.expectEqual(@as(?u32, 256), iter.next());
    try std.testing.expectEqual(@as(?u32, 512), iter.next());
    try std.testing.expectEqual(@as(?u32, 768), iter.next());
    try std.testing.expectEqual(@as(?u32, null), iter.next());
}

// ============================================================================
// GridStrideIterator — multiple calls to next() after exhaustion
// ============================================================================

test "GridStrideIterator stays null after exhaustion" {
    var iter = types.GridStrideIterator{
        .current = 0,
        .stride = 1,
        .end = 2,
    };
    try std.testing.expectEqual(@as(?u32, 0), iter.next());
    try std.testing.expectEqual(@as(?u32, 1), iter.next());
    try std.testing.expectEqual(@as(?u32, null), iter.next());
    try std.testing.expectEqual(@as(?u32, null), iter.next()); // still null
    try std.testing.expectEqual(@as(?u32, null), iter.next()); // still null
}

// ============================================================================
// GridStrideIterator — non-zero start (simulates thread 5 in a 256-thread grid)
// ============================================================================

test "GridStrideIterator non-zero start: thread 5 in 256-wide grid, n=520" {
    var iter = types.GridStrideIterator{
        .current = 5,
        .stride = 256,
        .end = 520,
    };
    try std.testing.expectEqual(@as(?u32, 5), iter.next());
    try std.testing.expectEqual(@as(?u32, 261), iter.next());
    try std.testing.expectEqual(@as(?u32, 517), iter.next());
    try std.testing.expectEqual(@as(?u32, null), iter.next());
}

// ============================================================================
// GridStrideIterator — stride=1 exhaustive small range
// ============================================================================

test "GridStrideIterator stride=1 sequential: 0..5" {
    var iter = types.GridStrideIterator{
        .current = 0,
        .stride = 1,
        .end = 5,
    };
    var count: u32 = 0;
    while (iter.next()) |idx| {
        try std.testing.expectEqual(count, idx);
        count += 1;
    }
    try std.testing.expectEqual(@as(u32, 5), count);
}

// ============================================================================
// GridStrideIterator struct layout
// ============================================================================

test "GridStrideIterator struct has expected fields" {
    const iter = types.GridStrideIterator{
        .current = 42,
        .stride = 256,
        .end = 1000,
    };
    try std.testing.expectEqual(@as(u32, 42), iter.current);
    try std.testing.expectEqual(@as(u32, 256), iter.stride);
    try std.testing.expectEqual(@as(u32, 1000), iter.end);
}
