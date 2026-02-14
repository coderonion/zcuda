/// zCUDA: cuFFT - Safe abstraction layer.
///
/// Layer 3: High-level wrappers for cuFFT, supporting 1D/2D/3D FFT plans
/// with C2C, R2C, C2R execution modes.
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");
const driver = @import("../driver/driver.zig");

pub const CufftError = result.CufftError;

/// FFT direction.
pub const Direction = enum {
    forward,
    inverse,

    fn toSys(self: Direction) i32 {
        return switch (self) {
            .forward => sys.CUFFT_FORWARD,
            .inverse => sys.CUFFT_INVERSE,
        };
    }
};

/// FFT type.
pub const FftType = enum {
    c2c_f32, // Complex-to-complex single
    c2c_f64, // Complex-to-complex double
    r2c_f32, // Real-to-complex single
    c2r_f32, // Complex-to-real single
    r2c_f64, // Real-to-complex double
    c2r_f64, // Complex-to-real double

    fn toSys(self: FftType) sys.cufftType {
        return switch (self) {
            .c2c_f32 => sys.CUFFT_C2C,
            .c2c_f64 => sys.CUFFT_Z2Z,
            .r2c_f32 => sys.CUFFT_R2C,
            .c2r_f32 => sys.CUFFT_C2R,
            .r2c_f64 => sys.CUFFT_D2Z,
            .c2r_f64 => sys.CUFFT_Z2D,
        };
    }
};

/// A cuFFT plan context.
pub const CufftPlan = struct {
    handle: sys.cufftHandle,
    fft_type: FftType,

    const Self = @This();

    /// Create a 1D FFT plan.
    pub fn plan1d(nx: i32, fft_type: FftType, batch: i32) CufftError!Self {
        const handle = try result.plan1d(nx, fft_type.toSys(), batch);
        return Self{ .handle = handle, .fft_type = fft_type };
    }

    /// Create a 2D FFT plan.
    pub fn plan2d(nx: i32, ny: i32, fft_type: FftType) CufftError!Self {
        const handle = try result.plan2d(nx, ny, fft_type.toSys());
        return Self{ .handle = handle, .fft_type = fft_type };
    }

    /// Create a 3D FFT plan.
    pub fn plan3d(nx: i32, ny: i32, nz: i32, fft_type: FftType) CufftError!Self {
        const handle = try result.plan3d(nx, ny, nz, fft_type.toSys());
        return Self{ .handle = handle, .fft_type = fft_type };
    }

    /// Create a multi-dimensional batched FFT plan (advanced layout).
    pub fn planMany(
        rank: c_int,
        n: [*c]c_int,
        inembed: ?[*c]c_int,
        istride: c_int,
        idist: c_int,
        onembed: ?[*c]c_int,
        ostride: c_int,
        odist: c_int,
        fft_type: FftType,
        batch: c_int,
    ) CufftError!Self {
        const handle = try result.planMany(rank, n, inembed, istride, idist, onembed, ostride, odist, fft_type.toSys(), batch);
        return Self{ .handle = handle, .fft_type = fft_type };
    }

    /// Destroy the FFT plan.
    pub fn deinit(self: Self) void {
        result.destroy(self.handle) catch {};
    }

    /// Get the workspace size needed for this plan (in bytes).
    pub fn getSize(self: Self) CufftError!usize {
        return result.getSize(self.handle);
    }

    /// Set the CUDA stream for this plan.
    pub fn setStream(self: Self, stream: *const driver.CudaStream) CufftError!void {
        try result.setStream(self.handle, stream.stream);
    }

    /// Execute complex-to-complex FFT (float).
    pub fn execC2C(self: Self, input: driver.CudaSlice(f32), output: driver.CudaSlice(f32), direction: Direction) CufftError!void {
        try result.execC2C(self.handle, @ptrFromInt(input.ptr), @ptrFromInt(output.ptr), direction.toSys());
    }

    /// Execute complex-to-complex FFT (double).
    pub fn execZ2Z(self: Self, input: driver.CudaSlice(f64), output: driver.CudaSlice(f64), direction: Direction) CufftError!void {
        try result.execZ2Z(self.handle, @ptrFromInt(input.ptr), @ptrFromInt(output.ptr), direction.toSys());
    }

    /// Execute real-to-complex FFT (float).
    pub fn execR2C(self: Self, input: driver.CudaSlice(f32), output: driver.CudaSlice(f32)) CufftError!void {
        try result.execR2C(self.handle, @ptrFromInt(input.ptr), @ptrFromInt(output.ptr));
    }

    /// Execute complex-to-real FFT (float).
    pub fn execC2R(self: Self, input: driver.CudaSlice(f32), output: driver.CudaSlice(f32)) CufftError!void {
        try result.execC2R(self.handle, @ptrFromInt(input.ptr), @ptrFromInt(output.ptr));
    }

    /// Execute real-to-complex FFT (double).
    pub fn execD2Z(self: Self, input: driver.CudaSlice(f64), output: driver.CudaSlice(f64)) CufftError!void {
        try result.execD2Z(self.handle, @ptrFromInt(input.ptr), @ptrFromInt(output.ptr));
    }

    /// Execute complex-to-real FFT (double).
    pub fn execZ2D(self: Self, input: driver.CudaSlice(f64), output: driver.CudaSlice(f64)) CufftError!void {
        try result.execZ2D(self.handle, @ptrFromInt(input.ptr), @ptrFromInt(output.ptr));
    }
};
