// src/kernel/arch.zig — GPU SM Architecture version definitions and comptime guards
//
// Provides:
//   - SmVersion enum for all supported NVIDIA SM architectures (sm_52 – sm_100+)
//   - requireSM() comptime guard for compile-time SM version checking
//   - comptimeIntToStr() helper for generating compile error messages
//
// Usage in intrinsics:
//   const arch = @import("arch.zig");
//   const SM = @import("device.zig").SM;
//
//   pub inline fn __reduce_add_sync(mask: u32, val: u32) u32 {
//       arch.requireSM(SM, .sm_80, "__reduce_add_sync()");
//       return asm volatile ("redux.sync.add.u32 %[ret], %[val], %[mask];"
//           : [ret] "=r" (-> u32)
//           : [val] "r" (val), [mask] "r" (mask)
//       );
//   }

/// Supported NVIDIA GPU SM architectures.
/// Non-exhaustive to allow forward compatibility with future architectures.
pub const SmVersion = enum(u32) {
    sm_52 = 52, // Maxwell
    sm_60 = 60, // Pascal
    sm_70 = 70, // Volta
    sm_75 = 75, // Turing
    sm_80 = 80, // Ampere (default)
    sm_86 = 86, // Ampere (consumer-grade)
    sm_89 = 89, // Ada Lovelace
    sm_90 = 90, // Hopper
    sm_100 = 100, // Blackwell
    _, // future architectures

    /// Returns the raw integer value of this SM version.
    pub fn asInt(self: SmVersion) u32 {
        return @intFromEnum(self);
    }

    /// Returns true if this SM version is at least `min`.
    pub fn atLeast(self: SmVersion, min: SmVersion) bool {
        return self.asInt() >= min.asInt();
    }

    /// Returns the architecture codename for display/debugging.
    pub fn codename(self: SmVersion) []const u8 {
        return switch (self) {
            .sm_52 => "Maxwell",
            .sm_60 => "Pascal",
            .sm_70 => "Volta",
            .sm_75 => "Turing",
            .sm_80 => "Ampere",
            .sm_86 => "Ampere (consumer)",
            .sm_89 => "Ada Lovelace",
            .sm_90 => "Hopper",
            .sm_100 => "Blackwell",
            _ => "Unknown",
        };
    }
};

/// Comptime SM guard — produces a clear compile error if the target SM
/// does not meet the minimum requirement for a given intrinsic/feature.
///
/// Example error output:
///   error: __reduce_add_sync() requires sm_80+, but target is sm_70
pub inline fn requireSM(
    comptime current: SmVersion,
    comptime minimum: SmVersion,
    comptime feature_name: []const u8,
) void {
    comptime {
        if (current.asInt() < minimum.asInt()) {
            @compileError(feature_name ++ " requires sm_" ++
                comptimeIntToStr(minimum.asInt()) ++
                "+, but target is sm_" ++
                comptimeIntToStr(current.asInt()));
        }
    }
}

/// Convert a comptime integer to a string for use in @compileError messages.
/// Handles values 0-999 (sufficient for SM version numbers).
pub fn comptimeIntToStr(comptime val: u32) *const [comptimeIntLen(val)]u8 {
    if (val == 0) return "0";

    comptime var buf: [10]u8 = undefined;
    comptime var len: usize = 0;
    comptime var v = val;

    comptime {
        while (v > 0) {
            buf[len] = @intCast('0' + (v % 10));
            len += 1;
            v /= 10;
        }
    }

    const result = comptime blk: {
        var r: [len]u8 = undefined;
        for (0..len) |i| {
            r[i] = buf[len - 1 - i];
        }
        break :blk r;
    };

    return &result;
}

fn comptimeIntLen(comptime val: u32) usize {
    if (val == 0) return 1;
    comptime var v = val;
    comptime var len: usize = 0;
    comptime {
        while (v > 0) {
            len += 1;
            v /= 10;
        }
    }
    return len;
}
