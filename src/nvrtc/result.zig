/// zCUDA: NVRTC API - Error wrapping layer.
///
/// Layer 2: Converts NVRTC C-style error codes to Zig error unions.
const std = @import("std");
const sys = @import("sys.zig");

// ============================================================================
// Error Type
// ============================================================================

/// Represents an NVRTC error.
pub const NvrtcError = error{
    OutOfMemory,
    ProgramCreationFailure,
    InvalidInput,
    InvalidProgram,
    InvalidOption,
    Compilation,
    BuiltinOperationFailure,
    NoNameExpressionsAfterCompilation,
    NoLoweredNamesBeforeCompilation,
    NameExpressionNotValid,
    InternalError,
};

/// Convert an nvrtcResult to a Zig error.
pub fn toError(result_code: sys.nvrtcResult) NvrtcError!void {
    return switch (result_code) {
        sys.NVRTC_SUCCESS => {},
        sys.NVRTC_ERROR_OUT_OF_MEMORY => NvrtcError.OutOfMemory,
        sys.NVRTC_ERROR_PROGRAM_CREATION_FAILURE => NvrtcError.ProgramCreationFailure,
        sys.NVRTC_ERROR_INVALID_INPUT => NvrtcError.InvalidInput,
        sys.NVRTC_ERROR_INVALID_PROGRAM => NvrtcError.InvalidProgram,
        sys.NVRTC_ERROR_INVALID_OPTION => NvrtcError.InvalidOption,
        sys.NVRTC_ERROR_COMPILATION => NvrtcError.Compilation,
        sys.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE => NvrtcError.BuiltinOperationFailure,
        else => NvrtcError.InternalError,
    };
}

/// Get a human-readable error string for an NVRTC result code.
pub fn getErrorString(result_code: sys.nvrtcResult) []const u8 {
    const str = sys.nvrtcGetErrorString(result_code);
    return std.mem.span(str);
}

// ============================================================================
// Version
// ============================================================================

/// NVRTC version information.
pub const NvrtcVersion = struct { major: i32, minor: i32 };

/// Get the NVRTC version.
pub fn getVersion() NvrtcError!NvrtcVersion {
    var major: i32 = undefined;
    var minor: i32 = undefined;
    try toError(sys.nvrtcVersion(&major, &minor));
    return .{ .major = major, .minor = minor };
}

// ============================================================================
// Program Management
// ============================================================================

/// Create a new NVRTC program from source code.
pub fn createProgram(
    src: [*:0]const u8,
    name: ?[*:0]const u8,
) NvrtcError!sys.nvrtcProgram {
    var prog: sys.nvrtcProgram = undefined;
    try toError(sys.nvrtcCreateProgram(
        &prog,
        src,
        name,
        0,
        null,
        null,
    ));
    return prog;
}

/// Destroy an NVRTC program and release its resources.
pub fn destroyProgram(prog: *sys.nvrtcProgram) NvrtcError!void {
    try toError(sys.nvrtcDestroyProgram(prog));
}

/// Compile a program with the given options.
pub fn compileProgram(prog: sys.nvrtcProgram, options: []const [*:0]const u8) NvrtcError!void {
    const num_options: i32 = @intCast(options.len);
    const options_ptr: [*]const [*:0]const u8 = options.ptr;
    try toError(sys.nvrtcCompileProgram(prog, num_options, options_ptr));
}

// ============================================================================
// PTX Output
// ============================================================================

/// Get the size of the compiled PTX in bytes.
pub fn getPTXSize(prog: sys.nvrtcProgram) NvrtcError!usize {
    var size: usize = undefined;
    try toError(sys.nvrtcGetPTXSize(prog, &size));
    return size;
}

/// Get the compiled PTX code.
pub fn getPTX(prog: sys.nvrtcProgram, ptx: []u8) NvrtcError!void {
    try toError(sys.nvrtcGetPTX(prog, ptx.ptr));
}

// ============================================================================
// Compilation Log
// ============================================================================

/// Get the size of the compilation log in bytes.
pub fn getProgramLogSize(prog: sys.nvrtcProgram) NvrtcError!usize {
    var size: usize = undefined;
    try toError(sys.nvrtcGetProgramLogSize(prog, &size));
    return size;
}

/// Get the compilation log (useful for debugging compilation errors).
pub fn getProgramLog(prog: sys.nvrtcProgram, log: []u8) NvrtcError!void {
    try toError(sys.nvrtcGetProgramLog(prog, log.ptr));
}

// ============================================================================
// CUBIN Output (JIT-compiled native code)
// ============================================================================

/// Get the size of the compiled CUBIN in bytes.
pub fn getCUBINSize(prog: sys.nvrtcProgram) NvrtcError!usize {
    var size: usize = undefined;
    try toError(sys.nvrtcGetCUBINSize(prog, &size));
    return size;
}

/// Get the compiled CUBIN (native GPU code, faster than PTX loading).
pub fn getCUBIN(prog: sys.nvrtcProgram, cubin: []u8) NvrtcError!void {
    try toError(sys.nvrtcGetCUBIN(prog, cubin.ptr));
}

// ============================================================================
// Named Expressions (template mangling resolution)
// ============================================================================

/// Register a name expression (e.g., C++ template instantiation) for lowered name lookup.
pub fn addNameExpression(prog: sys.nvrtcProgram, name_expression: [*:0]const u8) NvrtcError!void {
    try toError(sys.nvrtcAddNameExpression(prog, name_expression));
}

/// Get the lowered (mangled) name for a previously registered name expression.
/// Must be called after compilation. The returned pointer is valid until program is destroyed.
pub fn getLoweredName(prog: sys.nvrtcProgram, name_expression: [*:0]const u8) NvrtcError![*:0]const u8 {
    var lowered_name: [*c]const u8 = undefined;
    try toError(sys.nvrtcGetLoweredName(prog, name_expression, &lowered_name));
    return @ptrCast(lowered_name);
}

// ============================================================================
// Supported Architectures
// ============================================================================

/// Get the number of supported GPU architectures.
pub fn getNumSupportedArchs() NvrtcError!c_int {
    var num: c_int = undefined;
    try toError(sys.nvrtcGetNumSupportedArchs(&num));
    return num;
}

/// Get the list of supported GPU architectures (SM versions).
pub fn getSupportedArchs(archs: [*c]c_int) NvrtcError!void {
    try toError(sys.nvrtcGetSupportedArchs(archs));
}
