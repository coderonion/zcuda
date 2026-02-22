// src/kernel/debug.zig — Device-side debugging and error handling utilities
//
// Provides GPU-side safety mechanisms:
//   - assertf(): device-side assert that traps on failure
//   - bounds check helpers
//   - Error flag pattern for host-side error detection
//   - printf-equivalent for device-side debugging (via PTX vprintf)
//
// Usage:
//   const cuda = @import("zcuda_kernel");
//   const debug = cuda.debug;
//
//   export fn myKernel(data: [*]f32, n: u32) callconv(.kernel) void {
//       const i = cuda.types.globalThreadIdx();
//       debug.assertInBounds(i, n);
//       data[i] = 42.0;
//   }

const intrinsics = @import("intrinsics.zig");

// ============================================================================
// Device-side Trap / Breakpoint / Abort
// ============================================================================

/// __trap — terminates kernel execution, equivalent to CUDA C++ `__trap()`
/// All threads in the warp are halted. The host will receive an error.
pub inline fn __trap() noreturn {
    asm volatile ("trap;");
    unreachable;
}

/// __brkpt — triggers a breakpoint for debugger, equivalent to CUDA C++ `__brkpt()`
pub inline fn __brkpt() void {
    asm volatile ("brkpt;");
}

// ============================================================================
// Device-side Assertions
// ============================================================================

/// assertf — device-side assert. If `condition` is false, traps the kernel.
/// Equivalent to `assert()` in CUDA C++ device code.
///
/// Usage:
///   debug.assertf(i < n);
///
/// On failure: all threads in the warp halt, host receives CUDA error.
pub inline fn assertf(condition: bool) void {
    if (!condition) {
        __trap();
    }
}

// ============================================================================
// Bounds Checking Helpers
// ============================================================================

/// assertInBounds — checks that `idx < bound`, traps if out-of-bounds.
/// Use this before array accesses to catch buffer overflows on GPU.
pub inline fn assertInBounds(idx: u32, bound: u32) void {
    assertf(idx < bound);
}

/// safeGet — load from array with bounds check, returns default if OOB
pub inline fn safeGet(ptr: [*]const f32, idx: u32, len: u32, default: f32) f32 {
    if (idx >= len) return default;
    return ptr[idx];
}

// ============================================================================
// Error Flag Pattern — cooperative error detection between host and device
// ============================================================================

/// Error flag structure — allocate on device, check from host after kernel launch.
///
/// Usage pattern:
///   // Host: allocate error flag via cuMemAlloc, initialize to 0
///   // Kernel: use setError() to signal a problem
///   // Host: copy flag back and check after kernel launch
///
/// Example:
///   export fn myKernel(data: [*]f32, n: u32, err_flag: *ErrorFlag) callconv(.kernel) void {
///       const i = cuda.types.globalThreadIdx();
///       if (i >= n) {
///           debug.setError(err_flag, ErrorFlag.OUT_OF_BOUNDS);
///           return;
///       }
///       // ... normal processing ...
///   }
pub const ErrorFlag = extern struct {
    code: u32 = 0,

    // Error codes
    pub const NO_ERROR: u32 = 0;
    pub const OUT_OF_BOUNDS: u32 = 1;
    pub const NAN_DETECTED: u32 = 2;
    pub const INF_DETECTED: u32 = 3;
    pub const ASSERTION_FAILED: u32 = 4;
    pub const CUSTOM_ERROR: u32 = 0x100; // User-defined errors start here
};

/// Set error flag atomically (only first error is recorded)
pub inline fn setError(flag: *ErrorFlag, code: u32) void {
    _ = intrinsics.atomicCAS(&flag.code, ErrorFlag.NO_ERROR, code);
}

/// Check if NaN and set error flag
pub inline fn checkNaN(val: f32, flag: *ErrorFlag) void {
    // NaN != NaN is true
    if (val != val) {
        setError(flag, ErrorFlag.NAN_DETECTED);
    }
}

// ============================================================================
// Profiling Helpers
// ============================================================================

/// __prof_trigger — trigger a profiling event, equivalent to CUDA C++ `__prof_trigger()`
pub inline fn __prof_trigger(counter: u32) void {
    _ = counter;
    asm volatile ("pmevent 0;");
}

/// Measure elapsed cycles between two points using clock()
pub const CycleTimer = struct {
    start_cycle: u32,

    pub inline fn begin() CycleTimer {
        return .{ .start_cycle = intrinsics.clock() };
    }

    pub inline fn elapsed(self: CycleTimer) u32 {
        return intrinsics.clock() -% self.start_cycle;
    }
};

// ============================================================================
// Device-side printf — uses CUDA's vprintf runtime
// ============================================================================

/// CUDA device-side vprintf — provided by the CUDA runtime/driver.
/// Signature: int vprintf(const char* fmt, void* args)
/// `args` must point to a packed buffer of arguments on the stack.
extern fn vprintf(fmt: [*]const u8, args: *const anyopaque) callconv(.c) i32;

/// Device-side printf — Zig-ergonomic wrapper around CUDA vprintf.
///
/// Usage:
///   debug.printf("Thread %u val %f\n", .{ tid, val });
///
/// Mirrors CUDA C++ printf() but with Zig tuple syntax.
/// Each argument is packed into a local buffer matching CUDA's expected layout.
pub inline fn printf(comptime fmt: [*:0]const u8, args: anytype) void {
    const ArgsType = @TypeOf(args);
    const fields = @typeInfo(ArgsType).@"struct".fields;

    if (fields.len == 0) {
        // No arguments — pass null
        _ = vprintf(fmt, @ptrFromInt(0));
        return;
    }

    // Build a packed buffer of all arguments, each 8-byte aligned (CUDA convention)
    comptime var buf_size: usize = 0;
    inline for (fields) |f| {
        const size = @max(@sizeOf(f.type), 8); // CUDA aligns each arg to 8 bytes
        buf_size += size;
    }

    var buf: [buf_size]u8 align(8) = undefined;
    comptime var offset: usize = 0;

    inline for (fields) |f| {
        const val = @field(args, f.name);
        const size = @max(@sizeOf(f.type), 8);

        // Write the value at the current offset
        if (@sizeOf(f.type) <= 4) {
            // Promote 32-bit values: u32/i32 stay as-is, f32 → f64 for printf
            if (f.type == f32) {
                const promoted: f64 = @floatCast(val);
                @as(*align(1) f64, @ptrCast(buf[offset .. offset + 8])).* = promoted;
            } else {
                // Zero-fill then write the value
                @memset(buf[offset .. offset + size], 0);
                @as(*align(1) @TypeOf(val), @ptrCast(buf[offset .. offset + @sizeOf(f.type)])).* = val;
            }
        } else {
            @as(*align(1) @TypeOf(val), @ptrCast(buf[offset .. offset + @sizeOf(f.type)])).* = val;
        }

        offset += size;
    }

    _ = vprintf(fmt, @ptrCast(&buf));
}
