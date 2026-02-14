/// zCUDA: cuRAND API - Safe abstraction layer.
///
/// Layer 3: High-level API for GPU random number generation.
///
/// ## Example
///
/// ```zig
/// const rng = try CurandContext.init(cuda_ctx, .default);
/// defer rng.deinit();
/// rng.setSeed(42);
///
/// // Generate 1000 uniform random floats
/// const data = try stream.alloc(f32, allocator, 1000);
/// defer data.deinit();
/// try rng.fillUniform(f32, data);
/// ```
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");
const driver = @import("../driver/driver.zig");

pub const CurandError = result.CurandError;

/// Pseudo-random number generator type.
pub const RngType = enum {
    /// Default XORWOW generator.
    default,
    /// XORWOW generator.
    xorwow,
    /// MRG32k3a combined multiple recursive with 3 components.
    mrg32k3a,
    /// Mersenne Twister for GPU.
    mtgp32,
    /// Mersenne Twister 19937.
    mt19937,
    /// Philox 4x32 with 10 rounds.
    philox4_32_10,
    /// Sobol 32-bit quasi-random.
    sobol32,
    /// Scrambled Sobol 32-bit quasi-random.
    scrambled_sobol32,

    fn toSys(self: RngType) sys.curandRngType_t {
        return switch (self) {
            .default => sys.CURAND_RNG_PSEUDO_DEFAULT,
            .xorwow => sys.CURAND_RNG_PSEUDO_XORWOW,
            .mrg32k3a => sys.CURAND_RNG_PSEUDO_MRG32K3A,
            .mtgp32 => sys.CURAND_RNG_PSEUDO_MTGP32,
            .mt19937 => sys.CURAND_RNG_PSEUDO_MT19937,
            .philox4_32_10 => sys.CURAND_RNG_PSEUDO_PHILOX4_32_10,
            .sobol32 => sys.CURAND_RNG_QUASI_SOBOL32,
            .scrambled_sobol32 => sys.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
        };
    }
};

// ============================================================================
// CurandContext — cuRAND generator wrapper
// ============================================================================

/// A cuRAND context wrapping a pseudo-random number generator.
pub const CurandContext = struct {
    generator: sys.curandGenerator_t,
    cuda_ctx: *const driver.CudaContext,

    const Self = @This();

    /// Create a new cuRAND generator of the specified type.
    pub fn init(cuda_ctx: *const driver.CudaContext, rng_type: RngType) !Self {
        try cuda_ctx.bindToThread();
        const gen = try result.createGenerator(rng_type.toSys());
        return Self{
            .generator = gen,
            .cuda_ctx = cuda_ctx,
        };
    }

    /// Destroy the cuRAND generator.
    pub fn deinit(self: Self) void {
        result.destroyGenerator(self.generator) catch {};
    }

    /// Set the seed.
    pub fn setSeed(self: Self, seed: u64) CurandError!void {
        try result.setSeed(self.generator, seed);
    }

    /// Set the generator offset.
    pub fn setOffset(self: Self, offset: u64) CurandError!void {
        try result.setOffset(self.generator, offset);
    }

    /// Set the number of dimensions for quasi-random generators (Sobol).
    pub fn setDimensions(self: Self, num_dimensions: u32) CurandError!void {
        try result.setQuasiRandomGeneratorDimensions(self.generator, num_dimensions);
    }

    /// Fill device memory with uniform random floats on (0, 1].
    pub fn fillUniform(self: Self, data: driver.CudaSlice(f32)) CurandError!void {
        try result.generateUniform(self.generator, @ptrFromInt(data.ptr), data.len);
    }

    /// Fill device memory with uniform random doubles on (0, 1].
    pub fn fillUniformDouble(self: Self, data: driver.CudaSlice(f64)) CurandError!void {
        try result.generateUniformDouble(self.generator, @ptrFromInt(data.ptr), data.len);
    }

    /// Fill device memory with normally distributed random floats.
    pub fn fillNormal(self: Self, data: driver.CudaSlice(f32), mean: f32, stddev: f32) CurandError!void {
        try result.generateNormal(self.generator, @ptrFromInt(data.ptr), data.len, mean, stddev);
    }

    /// Fill device memory with normally distributed random doubles.
    pub fn fillNormalDouble(self: Self, data: driver.CudaSlice(f64), mean: f64, stddev: f64) CurandError!void {
        try result.generateNormalDouble(self.generator, @ptrFromInt(data.ptr), data.len, mean, stddev);
    }

    /// Fill device memory with log-normally distributed random floats.
    pub fn fillLogNormal(self: Self, data: driver.CudaSlice(f32), mean: f32, stddev: f32) CurandError!void {
        try result.generateLogNormal(self.generator, @ptrFromInt(data.ptr), data.len, mean, stddev);
    }

    /// Fill device memory with log-normally distributed random doubles.
    pub fn fillLogNormalDouble(self: Self, data: driver.CudaSlice(f64), mean: f64, stddev: f64) CurandError!void {
        try result.generateLogNormalDouble(self.generator, @ptrFromInt(data.ptr), data.len, mean, stddev);
    }

    /// Fill device memory with uniform random u32 values.
    pub fn fillUniformU32(self: Self, data: driver.CudaSlice(u32)) CurandError!void {
        try result.generate(self.generator, @ptrFromInt(data.ptr), data.len);
    }

    /// Set the CUDA stream for this generator.
    pub fn setStream(self: Self, stream: *const driver.CudaStream) CurandError!void {
        try result.setStream(self.generator, stream.stream);
    }

    /// Fill device memory with Poisson-distributed unsigned 32-bit integers.
    pub fn fillPoisson(self: Self, data: driver.CudaSlice(u32), lambda: f64) CurandError!void {
        try result.generatePoisson(self.generator, @ptrFromInt(data.ptr), data.len, lambda);
    }
};

// ============================================================================
// Tests
// ============================================================================
