/// zCUDA: cuBLAS LT API - Safe abstraction layer.
///
/// Layer 3: High-level, type-safe cuBLAS LT operations for advanced GEMM.
///
/// cuBLAS LT provides finer control over matrix multiplication,
/// including algorithm selection, workspace management, and mixed-precision.
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");
const driver = @import("../driver/driver.zig");

pub const CublasLtError = result.CublasLtError;

/// Data type for matrix elements.
pub const DataType = enum {
    f16,
    bf16,
    f32,
    f64,

    fn toSys(self: DataType) sys.cudaDataType {
        return switch (self) {
            .f16 => sys.CUDA_R_16F,
            .bf16 => sys.CUDA_R_16BF,
            .f32 => sys.CUDA_R_32F,
            .f64 => sys.CUDA_R_64F,
        };
    }
};

/// Compute type for matmul operations.
pub const ComputeType = enum {
    f16,
    f32,
    f64,
    f32_fast_tf32,

    fn toSys(self: ComputeType) sys.cublasComputeType_t {
        return switch (self) {
            .f16 => sys.CUBLAS_COMPUTE_16F,
            .f32 => sys.CUBLAS_COMPUTE_32F,
            .f64 => sys.CUBLAS_COMPUTE_64F,
            .f32_fast_tf32 => sys.CUBLAS_COMPUTE_32F_FAST_TF32,
        };
    }
};

/// Operation type for matrix transposition.
pub const Operation = enum {
    none,
    transpose,
    conjugate_transpose,

    fn toSys(self: Operation) sys.cublasOperation_t {
        return switch (self) {
            .none => sys.CUBLAS_OP_N,
            .transpose => sys.CUBLAS_OP_T,
            .conjugate_transpose => sys.CUBLAS_OP_C,
        };
    }
};

/// Epilogue (fused post-operation) for matmul.
pub const Epilogue = enum {
    default,
    relu,
    bias,
    relu_bias,
    gelu,
    gelu_bias,

    fn toSys(self: Epilogue) sys.cublasLtEpilogue_t {
        return switch (self) {
            .default => sys.CUBLASLT_EPILOGUE_DEFAULT,
            .relu => sys.CUBLASLT_EPILOGUE_RELU,
            .bias => sys.CUBLASLT_EPILOGUE_BIAS,
            .relu_bias => sys.CUBLASLT_EPILOGUE_RELU_BIAS,
            .gelu => sys.CUBLASLT_EPILOGUE_GELU,
            .gelu_bias => sys.CUBLASLT_EPILOGUE_GELU_BIAS,
        };
    }
};

// ============================================================================
// CublasLtContext
// ============================================================================

/// A cuBLAS LT context for advanced matrix multiplication.
pub const CublasLtContext = struct {
    handle: sys.cublasLtHandle_t,
    cuda_ctx: *const driver.CudaContext,

    const Self = @This();

    /// Create a new cuBLAS LT context.
    pub fn init(cuda_ctx: *const driver.CudaContext) !Self {
        try cuda_ctx.bindToThread();
        const handle = try result.create();
        return Self{
            .handle = handle,
            .cuda_ctx = cuda_ctx,
        };
    }

    /// Destroy the cuBLAS LT handle.
    pub fn deinit(self: Self) void {
        result.destroy(self.handle) catch {};
    }

    /// Create matrix multiplication descriptor.
    pub fn createMatmulDesc(self: Self, compute_type: ComputeType, scale_type: DataType) CublasLtError!sys.cublasLtMatmulDesc_t {
        _ = self;
        return result.matmulDescCreate(compute_type.toSys(), scale_type.toSys());
    }

    /// Create a matrix layout descriptor.
    pub fn createMatrixLayout(self: Self, data_type: DataType, rows: u64, cols: u64, ld: i64) CublasLtError!sys.cublasLtMatrixLayout_t {
        _ = self;
        return result.matrixLayoutCreate(data_type.toSys(), rows, cols, ld);
    }

    // --- Attribute Setters ---

    /// Set transpose operation for matrix A.
    pub fn setTransA(self: Self, desc: sys.cublasLtMatmulDesc_t, op: Operation) CublasLtError!void {
        _ = self;
        const op_val = op.toSys();
        try result.matmulDescSetAttribute(desc, sys.CUBLASLT_MATMUL_DESC_TRANSA, @ptrCast(&op_val), @sizeOf(sys.cublasOperation_t));
    }

    /// Set transpose operation for matrix B.
    pub fn setTransB(self: Self, desc: sys.cublasLtMatmulDesc_t, op: Operation) CublasLtError!void {
        _ = self;
        const op_val = op.toSys();
        try result.matmulDescSetAttribute(desc, sys.CUBLASLT_MATMUL_DESC_TRANSB, @ptrCast(&op_val), @sizeOf(sys.cublasOperation_t));
    }

    /// Set epilogue (fused post-op) on matmul descriptor.
    pub fn setEpilogue(self: Self, desc: sys.cublasLtMatmulDesc_t, epilogue: Epilogue) CublasLtError!void {
        _ = self;
        const ep_val = epilogue.toSys();
        try result.matmulDescSetAttribute(desc, sys.CUBLASLT_MATMUL_DESC_EPILOGUE, @ptrCast(&ep_val), @sizeOf(sys.cublasLtEpilogue_t));
    }

    /// Set max workspace bytes on a matmul preference.
    pub fn setMaxWorkspaceBytes(self: Self, pref: sys.cublasLtMatmulPreference_t, bytes: usize) CublasLtError!void {
        _ = self;
        try result.matmulPreferenceSetAttribute(pref, sys.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, @ptrCast(&bytes), @sizeOf(usize));
    }

    /// Set batch count on a matrix layout.
    pub fn setLayoutBatchCount(self: Self, layout: sys.cublasLtMatrixLayout_t, count: i32) CublasLtError!void {
        _ = self;
        try result.matrixLayoutSetAttribute(layout, sys.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, @ptrCast(&count), @sizeOf(i32));
    }

    /// Set strided batch offset on a matrix layout.
    pub fn setLayoutStridedBatchOffset(self: Self, layout: sys.cublasLtMatrixLayout_t, offset: i64) CublasLtError!void {
        _ = self;
        try result.matrixLayoutSetAttribute(layout, sys.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, @ptrCast(&offset), @sizeOf(i64));
    }

    // --- Heuristic & Algorithm Selection ---

    /// Create a matmul preference descriptor.
    pub fn createPreference(self: Self) CublasLtError!sys.cublasLtMatmulPreference_t {
        _ = self;
        return result.matmulPreferenceCreate();
    }

    /// Get multiple algorithm heuristics for benchmarking.
    /// Returns the number of valid results written to `results_buf`.
    pub fn getHeuristics(
        self: Self,
        matmul_desc: sys.cublasLtMatmulDesc_t,
        a_layout: sys.cublasLtMatrixLayout_t,
        b_layout: sys.cublasLtMatrixLayout_t,
        c_layout: sys.cublasLtMatrixLayout_t,
        d_layout: sys.cublasLtMatrixLayout_t,
        pref: sys.cublasLtMatmulPreference_t,
        results_buf: []sys.cublasLtMatmulHeuristicResult_t,
    ) CublasLtError!i32 {
        var return_count: i32 = 0;
        try result.matmulAlgoGetHeuristic(
            self.handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            d_layout,
            pref,
            @intCast(results_buf.len),
            results_buf.ptr,
            &return_count,
        );
        return return_count;
    }

    // --- Matmul Execution ---

    /// Perform matmul: D = alpha * op(A) * op(B) + beta * C
    /// Uses heuristic to automatically select the best algorithm.
    /// No workspace is used (simple mode).
    pub fn matmul(
        self: Self,
        comptime T: type,
        matmul_desc: sys.cublasLtMatmulDesc_t,
        alpha: T,
        a: driver.CudaSlice(T),
        a_layout: sys.cublasLtMatrixLayout_t,
        b: driver.CudaSlice(T),
        b_layout: sys.cublasLtMatrixLayout_t,
        beta: T,
        c: driver.CudaSlice(T),
        c_layout: sys.cublasLtMatrixLayout_t,
        d: driver.CudaSlice(T),
        d_layout: sys.cublasLtMatrixLayout_t,
        stream: ?*const driver.CudaStream,
    ) CublasLtError!void {
        const alpha_val = alpha;
        const beta_val = beta;

        // Get heuristic
        const pref = try result.matmulPreferenceCreate();
        defer result.matmulPreferenceDestroy(pref) catch {};

        var heuristic: sys.cublasLtMatmulHeuristicResult_t = undefined;
        var return_count: i32 = 0;
        try result.matmulAlgoGetHeuristic(
            self.handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            d_layout,
            pref,
            1,
            &heuristic,
            &return_count,
        );

        if (return_count == 0) return CublasLtError.NotSupported;

        const stream_ptr: ?*anyopaque = if (stream) |s| @ptrCast(s.stream) else null;

        try result.matmul(
            self.handle,
            matmul_desc,
            @ptrCast(&alpha_val),
            @ptrFromInt(a.ptr),
            a_layout,
            @ptrFromInt(b.ptr),
            b_layout,
            @ptrCast(&beta_val),
            @ptrFromInt(c.ptr),
            c_layout,
            @ptrFromInt(d.ptr),
            d_layout,
            &heuristic.algo,
            null,
            0,
            stream_ptr,
        );
    }

    /// Perform matmul with explicit algorithm and workspace.
    /// Use `getHeuristics()` to obtain algorithms and their workspace requirements.
    pub fn matmulWithAlgo(
        self: Self,
        comptime T: type,
        matmul_desc: sys.cublasLtMatmulDesc_t,
        alpha: T,
        a: driver.CudaSlice(T),
        a_layout: sys.cublasLtMatrixLayout_t,
        b: driver.CudaSlice(T),
        b_layout: sys.cublasLtMatrixLayout_t,
        beta: T,
        c: driver.CudaSlice(T),
        c_layout: sys.cublasLtMatrixLayout_t,
        d: driver.CudaSlice(T),
        d_layout: sys.cublasLtMatrixLayout_t,
        algo: *const sys.cublasLtMatmulAlgo_t,
        workspace: ?*anyopaque,
        workspace_size: usize,
        stream: ?*const driver.CudaStream,
    ) CublasLtError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        const stream_ptr: ?*anyopaque = if (stream) |s| @ptrCast(s.stream) else null;

        try result.matmul(
            self.handle,
            matmul_desc,
            @ptrCast(&alpha_val),
            @ptrFromInt(a.ptr),
            a_layout,
            @ptrFromInt(b.ptr),
            b_layout,
            @ptrCast(&beta_val),
            @ptrFromInt(c.ptr),
            c_layout,
            @ptrFromInt(d.ptr),
            d_layout,
            algo,
            workspace,
            workspace_size,
            stream_ptr,
        );
    }
};

/// Destroy a matmul descriptor.
pub fn destroyMatmulDesc(desc: sys.cublasLtMatmulDesc_t) void {
    result.matmulDescDestroy(desc) catch {};
}

/// Destroy a matrix layout descriptor.
pub fn destroyMatrixLayout(layout: sys.cublasLtMatrixLayout_t) void {
    result.matrixLayoutDestroy(layout) catch {};
}

/// Destroy a matmul preference.
pub fn destroyPreference(pref: sys.cublasLtMatmulPreference_t) void {
    result.matmulPreferenceDestroy(pref) catch {};
}

/// Re-exported heuristic result type for use in matmul algorithm selection.
pub const MatmulHeuristicResult = sys.cublasLtMatmulHeuristicResult_t;

// ============================================================================
// Tests
// ============================================================================
