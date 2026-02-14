/// zCUDA: cuFFT module — Fast Fourier Transform.
pub const sys = @import("sys.zig");
pub const result = @import("result.zig");
const safe = @import("safe.zig");

pub const CufftPlan = safe.CufftPlan;
pub const FftType = safe.FftType;
pub const CufftError = safe.CufftError;

test {
    _ = safe;
}
