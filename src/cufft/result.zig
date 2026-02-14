/// zCUDA: cuFFT - Error wrapping layer.
///
/// Layer 2: Converts cuFFT result codes to Zig error unions.
const sys = @import("sys.zig");

pub const CufftError = error{
    InvalidPlan,
    AllocFailed,
    InvalidType,
    InvalidValue,
    InternalError,
    ExecFailed,
    SetupFailed,
    InvalidSize,
    Incomplete,
    Unknown,
};

pub fn toError(status: sys.cufftResult) CufftError!void {
    return switch (status) {
        sys.CUFFT_SUCCESS => {},
        sys.CUFFT_INVALID_PLAN => CufftError.InvalidPlan,
        sys.CUFFT_ALLOC_FAILED => CufftError.AllocFailed,
        sys.CUFFT_INVALID_TYPE => CufftError.InvalidType,
        sys.CUFFT_INVALID_VALUE => CufftError.InvalidValue,
        sys.CUFFT_INTERNAL_ERROR => CufftError.InternalError,
        sys.CUFFT_EXEC_FAILED => CufftError.ExecFailed,
        sys.CUFFT_SETUP_FAILED => CufftError.SetupFailed,
        sys.CUFFT_INVALID_SIZE => CufftError.InvalidSize,
        sys.CUFFT_INCOMPLETE_PARAMETER_LIST => CufftError.Incomplete,
        else => CufftError.Unknown,
    };
}

// ============================================================================
// Plan Management
// ============================================================================

pub fn plan1d(nx: i32, fft_type: sys.cufftType, batch: i32) CufftError!sys.cufftHandle {
    var handle: sys.cufftHandle = undefined;
    try toError(sys.cufftPlan1d(&handle, nx, fft_type, batch));
    return handle;
}

pub fn plan2d(nx: i32, ny: i32, fft_type: sys.cufftType) CufftError!sys.cufftHandle {
    var handle: sys.cufftHandle = undefined;
    try toError(sys.cufftPlan2d(&handle, nx, ny, fft_type));
    return handle;
}

pub fn plan3d(nx: i32, ny: i32, nz: i32, fft_type: sys.cufftType) CufftError!sys.cufftHandle {
    var handle: sys.cufftHandle = undefined;
    try toError(sys.cufftPlan3d(&handle, nx, ny, nz, fft_type));
    return handle;
}

pub fn destroy(handle: sys.cufftHandle) CufftError!void {
    try toError(sys.cufftDestroy(handle));
}

pub fn setStream(handle: sys.cufftHandle, stream: ?*anyopaque) CufftError!void {
    try toError(sys.cufftSetStream(handle, @ptrCast(stream)));
}

/// Multi-dimensional batched FFT (most common production API).
/// n[] specifies dimensions, inembed/onembed control advanced data layout.
pub fn planMany(
    rank: c_int,
    n: [*c]c_int,
    inembed: ?[*c]c_int,
    istride: c_int,
    idist: c_int,
    onembed: ?[*c]c_int,
    ostride: c_int,
    odist: c_int,
    fft_type: sys.cufftType,
    batch: c_int,
) CufftError!sys.cufftHandle {
    var handle: sys.cufftHandle = undefined;
    try toError(sys.cufftPlanMany(&handle, rank, n, inembed orelse null, istride, idist, onembed orelse null, ostride, odist, fft_type, batch));
    return handle;
}

// ============================================================================
// Workspace Management
// ============================================================================

/// Get estimated workspace size for an existing plan.
pub fn getSize(handle: sys.cufftHandle) CufftError!usize {
    var size: usize = undefined;
    try toError(sys.cufftGetSize(handle, &size));
    return size;
}

/// Set a custom workspace area for the plan.
pub fn setWorkArea(handle: sys.cufftHandle, work_area: *anyopaque) CufftError!void {
    try toError(sys.cufftSetWorkArea(handle, work_area));
}

/// Disable/enable automatic workspace allocation.
/// If disabled, user must call setWorkArea() before execution.
pub fn setAutoAllocation(handle: sys.cufftHandle, auto_allocate: bool) CufftError!void {
    try toError(sys.cufftSetAutoAllocation(handle, if (auto_allocate) @as(c_int, 1) else @as(c_int, 0)));
}

// ============================================================================
// Execution
// ============================================================================

/// Complex-to-complex FFT (float).
pub fn execC2C(handle: sys.cufftHandle, idata: *anyopaque, odata: *anyopaque, direction: i32) CufftError!void {
    try toError(sys.cufftExecC2C(handle, @ptrCast(@alignCast(idata)), @ptrCast(@alignCast(odata)), direction));
}

/// Complex-to-complex FFT (double).
pub fn execZ2Z(handle: sys.cufftHandle, idata: *anyopaque, odata: *anyopaque, direction: i32) CufftError!void {
    try toError(sys.cufftExecZ2Z(handle, @ptrCast(@alignCast(idata)), @ptrCast(@alignCast(odata)), direction));
}

/// Real-to-complex FFT (float).
pub fn execR2C(handle: sys.cufftHandle, idata: *anyopaque, odata: *anyopaque) CufftError!void {
    try toError(sys.cufftExecR2C(handle, @ptrCast(@alignCast(idata)), @ptrCast(@alignCast(odata))));
}

/// Complex-to-real FFT (float).
pub fn execC2R(handle: sys.cufftHandle, idata: *anyopaque, odata: *anyopaque) CufftError!void {
    try toError(sys.cufftExecC2R(handle, @ptrCast(@alignCast(idata)), @ptrCast(@alignCast(odata))));
}

/// Real-to-complex FFT (double).
pub fn execD2Z(handle: sys.cufftHandle, idata: *anyopaque, odata: *anyopaque) CufftError!void {
    try toError(sys.cufftExecD2Z(handle, @ptrCast(@alignCast(idata)), @ptrCast(@alignCast(odata))));
}

/// Complex-to-real FFT (double).
pub fn execZ2D(handle: sys.cufftHandle, idata: *anyopaque, odata: *anyopaque) CufftError!void {
    try toError(sys.cufftExecZ2D(handle, @ptrCast(@alignCast(idata)), @ptrCast(@alignCast(odata))));
}
