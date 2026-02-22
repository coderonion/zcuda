/// zCUDA Unit Tests: Kernel arch module (SM version logic)
///
/// Tests SmVersion enum, comparison logic, codename lookup,
/// and comptimeIntToStr helper — all pure comptime Zig.
const std = @import("std");
const arch = @import("arch");

// ============================================================================
// SmVersion.asInt
// ============================================================================

test "SmVersion.asInt returns raw integer" {
    try std.testing.expectEqual(@as(u32, 70), arch.SmVersion.sm_70.asInt());
    try std.testing.expectEqual(@as(u32, 80), arch.SmVersion.sm_80.asInt());
    try std.testing.expectEqual(@as(u32, 90), arch.SmVersion.sm_90.asInt());
    try std.testing.expectEqual(@as(u32, 100), arch.SmVersion.sm_100.asInt());
}

// ============================================================================
// SmVersion.atLeast
// ============================================================================

test "atLeast — same version" {
    try std.testing.expect(arch.SmVersion.sm_70.atLeast(.sm_70));
    try std.testing.expect(arch.SmVersion.sm_80.atLeast(.sm_80));
}

test "atLeast — higher version" {
    try std.testing.expect(arch.SmVersion.sm_80.atLeast(.sm_70));
    try std.testing.expect(arch.SmVersion.sm_90.atLeast(.sm_52));
    try std.testing.expect(arch.SmVersion.sm_100.atLeast(.sm_90));
}

test "atLeast — lower version returns false" {
    try std.testing.expect(!arch.SmVersion.sm_70.atLeast(.sm_80));
    try std.testing.expect(!arch.SmVersion.sm_52.atLeast(.sm_90));
    try std.testing.expect(!arch.SmVersion.sm_89.atLeast(.sm_90));
}

test "atLeast — edge cases" {
    // sm_86 vs sm_89 (same generation, different variant)
    try std.testing.expect(!arch.SmVersion.sm_86.atLeast(.sm_89));
    try std.testing.expect(arch.SmVersion.sm_89.atLeast(.sm_86));
}

// ============================================================================
// SmVersion.codename
// ============================================================================

test "codename returns correct architecture names" {
    try std.testing.expectEqualStrings("Maxwell", arch.SmVersion.sm_52.codename());
    try std.testing.expectEqualStrings("Pascal", arch.SmVersion.sm_60.codename());
    try std.testing.expectEqualStrings("Volta", arch.SmVersion.sm_70.codename());
    try std.testing.expectEqualStrings("Turing", arch.SmVersion.sm_75.codename());
    try std.testing.expectEqualStrings("Ampere", arch.SmVersion.sm_80.codename());
    try std.testing.expectEqualStrings("Ampere (consumer)", arch.SmVersion.sm_86.codename());
    try std.testing.expectEqualStrings("Ada Lovelace", arch.SmVersion.sm_89.codename());
    try std.testing.expectEqualStrings("Hopper", arch.SmVersion.sm_90.codename());
    try std.testing.expectEqualStrings("Blackwell", arch.SmVersion.sm_100.codename());
}

test "codename for unknown SM version" {
    const future: arch.SmVersion = @enumFromInt(120);
    try std.testing.expectEqualStrings("Unknown", future.codename());
}

// ============================================================================
// SmVersion enum from integer
// ============================================================================

test "SmVersion from integer round-trip" {
    const v: arch.SmVersion = @enumFromInt(80);
    try std.testing.expectEqual(arch.SmVersion.sm_80, v);
    try std.testing.expectEqual(@as(u32, 80), v.asInt());
}

test "SmVersion non-standard integer" {
    // Future SM version not in enum
    const v: arch.SmVersion = @enumFromInt(95);
    try std.testing.expectEqual(@as(u32, 95), v.asInt());
    try std.testing.expectEqualStrings("Unknown", v.codename());
}

// ============================================================================
// comptimeIntToStr
// ============================================================================

test "comptimeIntToStr single digit" {
    try std.testing.expectEqualStrings("0", arch.comptimeIntToStr(0));
    try std.testing.expectEqualStrings("5", arch.comptimeIntToStr(5));
    try std.testing.expectEqualStrings("9", arch.comptimeIntToStr(9));
}

test "comptimeIntToStr two digits" {
    try std.testing.expectEqualStrings("70", arch.comptimeIntToStr(70));
    try std.testing.expectEqualStrings("80", arch.comptimeIntToStr(80));
    try std.testing.expectEqualStrings("89", arch.comptimeIntToStr(89));
}

test "comptimeIntToStr three digits" {
    try std.testing.expectEqualStrings("100", arch.comptimeIntToStr(100));
    try std.testing.expectEqualStrings("120", arch.comptimeIntToStr(120));
    try std.testing.expectEqualStrings("999", arch.comptimeIntToStr(999));
}

// ============================================================================
// SM Version Ordering (comprehensive)
// ============================================================================

test "SM version total ordering" {
    const versions = [_]arch.SmVersion{
        .sm_52, .sm_60, .sm_70, .sm_75, .sm_80, .sm_86, .sm_89, .sm_90, .sm_100,
    };
    // Each version should be >= all previous versions
    for (versions, 0..) |v, i| {
        for (versions[0..i]) |prev| {
            try std.testing.expect(v.atLeast(prev));
        }
    }
    // Each version should NOT be >= any later version (strict ordering)
    for (versions, 0..) |v, i| {
        for (versions[i + 1 ..]) |later| {
            try std.testing.expect(!v.atLeast(later));
        }
    }
}
