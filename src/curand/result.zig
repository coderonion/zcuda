/// zCUDA: cuRAND API - Error wrapping layer.
///
/// Layer 2: Converts cuRAND C-style status codes to Zig error unions.
const std = @import("std");
const sys = @import("sys.zig");

// ============================================================================
// Error Type
// ============================================================================

/// Represents a cuRAND error.
pub const CurandError = error{
    VersionMismatch,
    NotInitialized,
    AllocationFailed,
    TypeError,
    OutOfRange,
    LengthNotMultiple,
    DoublePrecisionRequired,
    LaunchFailure,
    PreexistingFailure,
    InitializationFailed,
    ArchMismatch,
    InternalError,
    Unknown,
};

/// Convert a curandStatus_t to a Zig error.
pub fn toError(status: sys.curandStatus_t) CurandError!void {
    return switch (status) {
        sys.CURAND_STATUS_SUCCESS => {},
        sys.CURAND_STATUS_VERSION_MISMATCH => CurandError.VersionMismatch,
        sys.CURAND_STATUS_NOT_INITIALIZED => CurandError.NotInitialized,
        sys.CURAND_STATUS_ALLOCATION_FAILED => CurandError.AllocationFailed,
        sys.CURAND_STATUS_TYPE_ERROR => CurandError.TypeError,
        sys.CURAND_STATUS_OUT_OF_RANGE => CurandError.OutOfRange,
        sys.CURAND_STATUS_LENGTH_NOT_MULTIPLE => CurandError.LengthNotMultiple,
        sys.CURAND_STATUS_DOUBLE_PRECISION_REQUIRED => CurandError.DoublePrecisionRequired,
        sys.CURAND_STATUS_LAUNCH_FAILURE => CurandError.LaunchFailure,
        sys.CURAND_STATUS_PREEXISTING_FAILURE => CurandError.PreexistingFailure,
        sys.CURAND_STATUS_INITIALIZATION_FAILED => CurandError.InitializationFailed,
        sys.CURAND_STATUS_ARCH_MISMATCH => CurandError.ArchMismatch,
        sys.CURAND_STATUS_INTERNAL_ERROR => CurandError.InternalError,
        else => CurandError.Unknown,
    };
}

// ============================================================================
// Generator Management
// ============================================================================

/// Create a pseudo-random number generator.
pub fn createGenerator(rng_type: sys.curandRngType_t) CurandError!sys.curandGenerator_t {
    var gen: sys.curandGenerator_t = undefined;
    try toError(sys.curandCreateGenerator(&gen, rng_type));
    return gen;
}

/// Destroy a generator.
pub fn destroyGenerator(gen: sys.curandGenerator_t) CurandError!void {
    try toError(sys.curandDestroyGenerator(gen));
}

/// Set the seed for a pseudo-random number generator.
pub fn setSeed(gen: sys.curandGenerator_t, seed: u64) CurandError!void {
    try toError(sys.curandSetPseudoRandomGeneratorSeed(gen, seed));
}

/// Set the stream for a generator.
pub fn setStream(gen: sys.curandGenerator_t, stream: ?*anyopaque) CurandError!void {
    try toError(sys.curandSetStream(gen, @ptrCast(stream)));
}

/// Set the offset for a generator.
pub fn setOffset(gen: sys.curandGenerator_t, offset: u64) CurandError!void {
    try toError(sys.curandSetGeneratorOffset(gen, offset));
}

// ============================================================================
// Generation Functions
// ============================================================================

/// Generate uniformly distributed floats on (0, 1].
pub fn generateUniform(gen: sys.curandGenerator_t, output: [*c]f32, n: usize) CurandError!void {
    try toError(sys.curandGenerateUniform(gen, output, n));
}

/// Generate uniformly distributed doubles on (0, 1].
pub fn generateUniformDouble(gen: sys.curandGenerator_t, output: [*c]f64, n: usize) CurandError!void {
    try toError(sys.curandGenerateUniformDouble(gen, output, n));
}

/// Generate normally distributed floats with given mean and stddev.
pub fn generateNormal(gen: sys.curandGenerator_t, output: [*c]f32, n: usize, mean: f32, stddev: f32) CurandError!void {
    try toError(sys.curandGenerateNormal(gen, output, n, mean, stddev));
}

/// Generate normally distributed doubles with given mean and stddev.
pub fn generateNormalDouble(gen: sys.curandGenerator_t, output: [*c]f64, n: usize, mean: f64, stddev: f64) CurandError!void {
    try toError(sys.curandGenerateNormalDouble(gen, output, n, mean, stddev));
}

/// Generate 32-bit unsigned integers.
pub fn generate(gen: sys.curandGenerator_t, output: [*c]u32, n: usize) CurandError!void {
    try toError(sys.curandGenerate(gen, output, n));
}

/// Generate log-normally distributed floats.
pub fn generateLogNormal(gen: sys.curandGenerator_t, output: [*c]f32, n: usize, mean: f32, stddev: f32) CurandError!void {
    try toError(sys.curandGenerateLogNormal(gen, output, n, mean, stddev));
}

/// Generate log-normally distributed doubles.
pub fn generateLogNormalDouble(gen: sys.curandGenerator_t, output: [*c]f64, n: usize, mean: f64, stddev: f64) CurandError!void {
    try toError(sys.curandGenerateLogNormalDouble(gen, output, n, mean, stddev));
}

/// Set the number of dimensions for quasi-random generators (e.g., Sobol).
pub fn setQuasiRandomGeneratorDimensions(gen: sys.curandGenerator_t, num_dimensions: u32) CurandError!void {
    try toError(sys.curandSetQuasiRandomGeneratorDimensions(gen, @intCast(num_dimensions)));
}

/// Generate Poisson-distributed unsigned 32-bit integers.
pub fn generatePoisson(gen: sys.curandGenerator_t, output: [*c]u32, n: usize, lambda: f64) CurandError!void {
    try toError(sys.curandGeneratePoisson(gen, output, n, lambda));
}
