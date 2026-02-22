/// zCUDA Unit Tests: debug.zig struct layout and constants
///
/// Tests ErrorFlag struct, error code constants, CycleTimer type,
/// and function declarations. Runtime behavior (PTX asm) is untestable
/// on host — only comptime properties are verified.
const std = @import("std");
const debug = @import("debug");

// ============================================================================
// ErrorFlag — struct layout
// ============================================================================

test "ErrorFlag has code field" {
    const info = @typeInfo(debug.ErrorFlag);
    try std.testing.expect(info == .@"struct");
    const fields = info.@"struct".fields;
    try std.testing.expectEqual(@as(usize, 1), fields.len);
    try std.testing.expectEqualStrings("code", fields[0].name);
}

test "ErrorFlag is an extern struct" {
    const info = @typeInfo(debug.ErrorFlag);
    try std.testing.expectEqual(std.builtin.Type.ContainerLayout.@"extern", info.@"struct".layout);
}

test "ErrorFlag default is NO_ERROR" {
    const flag = debug.ErrorFlag{};
    try std.testing.expectEqual(@as(u32, 0), flag.code);
}

// ============================================================================
// ErrorFlag — error code constants
// ============================================================================

test "ErrorFlag.NO_ERROR is 0" {
    try std.testing.expectEqual(@as(u32, 0), debug.ErrorFlag.NO_ERROR);
}

test "ErrorFlag.OUT_OF_BOUNDS is 1" {
    try std.testing.expectEqual(@as(u32, 1), debug.ErrorFlag.OUT_OF_BOUNDS);
}

test "ErrorFlag.NAN_DETECTED is 2" {
    try std.testing.expectEqual(@as(u32, 2), debug.ErrorFlag.NAN_DETECTED);
}

test "ErrorFlag.INF_DETECTED is 3" {
    try std.testing.expectEqual(@as(u32, 3), debug.ErrorFlag.INF_DETECTED);
}

test "ErrorFlag.ASSERTION_FAILED is 4" {
    try std.testing.expectEqual(@as(u32, 4), debug.ErrorFlag.ASSERTION_FAILED);
}

test "ErrorFlag.CUSTOM_ERROR starts at 0x100" {
    try std.testing.expectEqual(@as(u32, 0x100), debug.ErrorFlag.CUSTOM_ERROR);
}

// ============================================================================
// ErrorFlag — all error codes are distinct
// ============================================================================

test "ErrorFlag error codes are all unique" {
    const codes = [_]u32{
        debug.ErrorFlag.NO_ERROR,
        debug.ErrorFlag.OUT_OF_BOUNDS,
        debug.ErrorFlag.NAN_DETECTED,
        debug.ErrorFlag.INF_DETECTED,
        debug.ErrorFlag.ASSERTION_FAILED,
        debug.ErrorFlag.CUSTOM_ERROR,
    };
    // Check no duplicates
    for (codes, 0..) |c, i| {
        for (codes[i + 1 ..]) |other| {
            try std.testing.expect(c != other);
        }
    }
}

// ============================================================================
// ErrorFlag — size
// ============================================================================

test "ErrorFlag is 4 bytes (1 × u32)" {
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(debug.ErrorFlag));
}

// ============================================================================
// CycleTimer — struct layout
// ============================================================================

test "CycleTimer has start_cycle field" {
    const info = @typeInfo(debug.CycleTimer);
    try std.testing.expect(info == .@"struct");
    const fields = info.@"struct".fields;
    try std.testing.expectEqual(@as(usize, 1), fields.len);
    try std.testing.expectEqualStrings("start_cycle", fields[0].name);
}

test "CycleTimer has begin and elapsed methods" {
    try std.testing.expect(@hasDecl(debug.CycleTimer, "begin"));
    try std.testing.expect(@hasDecl(debug.CycleTimer, "elapsed"));
}

// ============================================================================
// Function declarations existence
// ============================================================================

test "debug module exports assertf" {
    try std.testing.expect(@hasDecl(debug, "assertf"));
}

test "debug module exports assertInBounds" {
    try std.testing.expect(@hasDecl(debug, "assertInBounds"));
}

test "debug module exports safeGet" {
    try std.testing.expect(@hasDecl(debug, "safeGet"));
}

test "debug module exports setError" {
    try std.testing.expect(@hasDecl(debug, "setError"));
}

test "debug module exports checkNaN" {
    try std.testing.expect(@hasDecl(debug, "checkNaN"));
}

test "debug module exports __trap" {
    try std.testing.expect(@hasDecl(debug, "__trap"));
}

test "debug module exports __brkpt" {
    try std.testing.expect(@hasDecl(debug, "__brkpt"));
}

test "debug module exports __prof_trigger" {
    try std.testing.expect(@hasDecl(debug, "__prof_trigger"));
}
