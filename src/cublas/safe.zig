/// zCUDA: cuBLAS API - Safe abstraction layer.
///
/// Layer 3: High-level, type-safe cuBLAS operations.
///
/// ## Example
///
/// ```zig
/// const cublas_ctx = try CublasContext.init(cuda_ctx, stream);
/// defer cublas_ctx.deinit();
///
/// // SGEMM: C = A * B
/// try cublas_ctx.sgemm(.no_transpose, .no_transpose, m, n, k, 1.0, a, b, 0.0, c);
/// ```
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");
const driver = @import("../driver/driver.zig");

pub const CublasError = result.CublasError;

// ============================================================================
// Operation conversion helpers
// ============================================================================

fn toOp(op: Operation) sys.cublasOperation_t {
    return switch (op) {
        .no_transpose => sys.CUBLAS_OP_N,
        .transpose => sys.CUBLAS_OP_T,
        .conj_transpose => sys.CUBLAS_OP_C,
    };
}

/// cuBLAS transpose operation.
pub const Operation = enum {
    no_transpose,
    transpose,
    conj_transpose,
};

/// CUDA data type for mixed-precision operations.
pub const DataType = enum {
    f16,
    bf16,
    f32,
    f64,

    pub fn toSys(self: DataType) sys.cudaDataType_t {
        return switch (self) {
            .f16 => sys.CUDA_R_16F,
            .bf16 => sys.CUDA_R_16BF,
            .f32 => sys.CUDA_R_32F,
            .f64 => sys.CUDA_R_64F,
        };
    }
};

// ============================================================================
// CublasContext — cuBLAS handle wrapper
// ============================================================================

/// A cuBLAS context that wraps a cuBLAS handle.
///
/// The handle is associated with a CUDA stream through `setStream()`.
pub const CublasContext = struct {
    handle: sys.cublasHandle_t,
    cuda_ctx: *const driver.CudaContext,

    const Self = @This();

    /// Create a new cuBLAS context associated with a CUDA context.
    pub fn init(cuda_ctx: *const driver.CudaContext) !Self {
        try cuda_ctx.bindToThread();
        const handle = try result.create();
        return Self{
            .handle = handle,
            .cuda_ctx = cuda_ctx,
        };
    }

    /// Destroy the cuBLAS handle.
    pub fn deinit(self: Self) void {
        result.destroy(self.handle) catch {};
    }

    /// Set the stream for this cuBLAS handle.
    pub fn setStream(self: Self, stream: *const driver.CudaStream) CublasError!void {
        try result.toError(sys.c.cublasSetStream(self.handle, stream.stream));
    }

    /// Set pointer mode to HOST (scalar arguments passed as host pointers).
    pub fn setPointerModeHost(self: Self) CublasError!void {
        try result.setPointerMode(self.handle, sys.CUBLAS_POINTER_MODE_HOST);
    }

    /// Set pointer mode to DEVICE (scalar arguments passed as device pointers).
    pub fn setPointerModeDevice(self: Self) CublasError!void {
        try result.setPointerMode(self.handle, sys.CUBLAS_POINTER_MODE_DEVICE);
    }

    // --- BLAS Level 1 ---

    /// SAXPY: y = alpha * x + y (float).
    pub fn saxpy(
        self: Self,
        n: i32,
        alpha: f32,
        x: driver.CudaSlice(f32),
        y: driver.CudaSlice(f32),
    ) CublasError!void {
        const alpha_val = alpha;
        try result.saxpy(
            self.handle,
            n,
            &alpha_val,
            @ptrFromInt(x.ptr),
            1,
            @ptrFromInt(y.ptr),
            1,
        );
    }

    /// SASUM: Sum of absolute values (float). Returns the result.
    pub fn sasum(self: Self, n: i32, x: driver.CudaSlice(f32)) CublasError!f32 {
        var res: f32 = 0;
        try result.sasum(self.handle, n, @ptrFromInt(x.ptr), 1, &res);
        return res;
    }

    /// SNRM2: Euclidean norm (float). Returns the result.
    pub fn snrm2(self: Self, n: i32, x: driver.CudaSlice(f32)) CublasError!f32 {
        var res: f32 = 0;
        try result.snrm2(self.handle, n, @ptrFromInt(x.ptr), 1, &res);
        return res;
    }

    /// SSCAL: x = alpha * x (float).
    pub fn sscal(self: Self, n: i32, alpha: f32, x: driver.CudaSlice(f32)) CublasError!void {
        const alpha_val = alpha;
        try result.sscal(self.handle, n, &alpha_val, @ptrFromInt(x.ptr), 1);
    }

    // --- BLAS Level 2 ---

    /// SGEMV: y = alpha * op(A) * x + beta * y (float).
    pub fn sgemv(
        self: Self,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: f32,
        a: driver.CudaSlice(f32),
        lda: i32,
        x: driver.CudaSlice(f32),
        beta: f32,
        y: driver.CudaSlice(f32),
    ) CublasError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.sgemv(
            self.handle,
            toOp(trans),
            m,
            n,
            &alpha_val,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(x.ptr),
            1,
            &beta_val,
            @ptrFromInt(y.ptr),
            1,
        );
    }

    // --- BLAS Level 1: Givens Rotation ---

    /// Single-precision Givens rotation: x = c*x + s*y, y = -s*x + c*y.
    pub fn srot(self: Self, n: i32, x: anytype, incx: i32, y: anytype, incy: i32, c_val: f32, s: f32) CublasError!void {
        try result.srot(self.handle, n, @ptrFromInt(x.ptr), incx, @ptrFromInt(y.ptr), incy, &c_val, &s);
    }

    /// Double-precision Givens rotation: x = c*x + s*y, y = -s*x + c*y.
    pub fn drot(self: Self, n: i32, x: anytype, incx: i32, y: anytype, incy: i32, c_val: f64, s: f64) CublasError!void {
        try result.drot(self.handle, n, @ptrFromInt(x.ptr), incx, @ptrFromInt(y.ptr), incy, &c_val, &s);
    }

    // --- BLAS Level 2: Symmetric Matrix-Vector ---

    /// Single-precision SYMV: y = alpha * A * x + beta * y (A is symmetric).
    pub fn ssymv(self: Self, uplo: FillMode, n: i32, alpha: f32, a: anytype, lda: i32, x: anytype, incx: i32, beta: f32, y: anytype, incy: i32) CublasError!void {
        try result.ssymv(self.handle, uplo.toSys(), n, &alpha, @ptrFromInt(a.ptr), lda, @ptrFromInt(x.ptr), incx, &beta, @ptrFromInt(y.ptr), incy);
    }

    /// Double-precision SYMV: y = alpha * A * x + beta * y (A is symmetric).
    pub fn dsymv(self: Self, uplo: FillMode, n: i32, alpha: f64, a: anytype, lda: i32, x: anytype, incx: i32, beta: f64, y: anytype, incy: i32) CublasError!void {
        try result.dsymv(self.handle, uplo.toSys(), n, &alpha, @ptrFromInt(a.ptr), lda, @ptrFromInt(x.ptr), incx, &beta, @ptrFromInt(y.ptr), incy);
    }

    /// Single-precision SYR: A = alpha * x * x^T + A (symmetric rank-1 update).
    pub fn ssyr(self: Self, uplo: FillMode, n: i32, alpha: f32, x: anytype, incx: i32, a: anytype, lda: i32) CublasError!void {
        try result.ssyr(self.handle, uplo.toSys(), n, &alpha, @ptrFromInt(x.ptr), incx, @ptrFromInt(a.ptr), lda);
    }

    /// Double-precision SYR: A = alpha * x * x^T + A (symmetric rank-1 update).
    pub fn dsyr(self: Self, uplo: FillMode, n: i32, alpha: f64, x: anytype, incx: i32, a: anytype, lda: i32) CublasError!void {
        try result.dsyr(self.handle, uplo.toSys(), n, &alpha, @ptrFromInt(x.ptr), incx, @ptrFromInt(a.ptr), lda);
    }

    // --- BLAS Level 2: Triangular Ops ---

    /// Single-precision TRMV: x = op(A) * x (A is triangular).
    pub fn strmv(self: Self, uplo: FillMode, trans: Operation, diag: DiagType, n: i32, a: anytype, lda: i32, x: anytype, incx: i32) CublasError!void {
        try result.strmv(self.handle, uplo.toSys(), toOp(trans), diag.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(x.ptr), incx);
    }

    /// Double-precision TRMV: x = op(A) * x (A is triangular).
    pub fn dtrmv(self: Self, uplo: FillMode, trans: Operation, diag: DiagType, n: i32, a: anytype, lda: i32, x: anytype, incx: i32) CublasError!void {
        try result.dtrmv(self.handle, uplo.toSys(), toOp(trans), diag.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(x.ptr), incx);
    }

    /// Single-precision TRSV: solve op(A) * x = b for x (A is triangular, x overwrites b).
    pub fn strsv(self: Self, uplo: FillMode, trans: Operation, diag: DiagType, n: i32, a: anytype, lda: i32, x: anytype, incx: i32) CublasError!void {
        try result.strsv(self.handle, uplo.toSys(), toOp(trans), diag.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(x.ptr), incx);
    }

    /// Double-precision TRSV: solve op(A) * x = b for x (A is triangular, x overwrites b).
    pub fn dtrsv(self: Self, uplo: FillMode, trans: Operation, diag: DiagType, n: i32, a: anytype, lda: i32, x: anytype, incx: i32) CublasError!void {
        try result.dtrsv(self.handle, uplo.toSys(), toOp(trans), diag.toSys(), n, @ptrFromInt(a.ptr), lda, @ptrFromInt(x.ptr), incx);
    }

    // --- BLAS Level 3 ---

    /// SGEMM: C = alpha * op(A) * op(B) + beta * C (float).
    ///
    /// Performs general matrix-matrix multiplication.
    /// Note: cuBLAS uses column-major order.
    pub fn sgemm(
        self: Self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: driver.CudaSlice(f32),
        lda: i32,
        b: driver.CudaSlice(f32),
        ldb: i32,
        beta: f32,
        c_out: driver.CudaSlice(f32),
        ldc: i32,
    ) CublasError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.sgemm(
            self.handle,
            toOp(transa),
            toOp(transb),
            m,
            n,
            k,
            &alpha_val,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(b.ptr),
            ldb,
            &beta_val,
            @ptrFromInt(c_out.ptr),
            ldc,
        );
    }

    /// DGEMM: C = alpha * op(A) * op(B) + beta * C (double).
    pub fn dgemm(
        self: Self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: driver.CudaSlice(f64),
        lda: i32,
        b: driver.CudaSlice(f64),
        ldb: i32,
        beta: f64,
        c_out: driver.CudaSlice(f64),
        ldc: i32,
    ) CublasError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.dgemm(
            self.handle,
            toOp(transa),
            toOp(transb),
            m,
            n,
            k,
            &alpha_val,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(b.ptr),
            ldb,
            &beta_val,
            @ptrFromInt(c_out.ptr),
            ldc,
        );
    }

    // --- Double Precision Level 1 ---

    /// DAXPY: y = alpha * x + y (double).
    pub fn daxpy(
        self: Self,
        n: i32,
        alpha: f64,
        x: driver.CudaSlice(f64),
        y: driver.CudaSlice(f64),
    ) CublasError!void {
        const alpha_val = alpha;
        try result.daxpy(self.handle, n, &alpha_val, @ptrFromInt(x.ptr), 1, @ptrFromInt(y.ptr), 1);
    }

    /// DASUM: Sum of absolute values (double).
    pub fn dasum(self: Self, n: i32, x: driver.CudaSlice(f64)) CublasError!f64 {
        var res: f64 = 0;
        try result.dasum(self.handle, n, @ptrFromInt(x.ptr), 1, &res);
        return res;
    }

    /// DNRM2: Euclidean norm (double).
    pub fn dnrm2(self: Self, n: i32, x: driver.CudaSlice(f64)) CublasError!f64 {
        var res: f64 = 0;
        try result.dnrm2(self.handle, n, @ptrFromInt(x.ptr), 1, &res);
        return res;
    }

    /// DSCAL: x = alpha * x (double).
    pub fn dscal(self: Self, n: i32, alpha: f64, x: driver.CudaSlice(f64)) CublasError!void {
        const alpha_val = alpha;
        try result.dscal(self.handle, n, &alpha_val, @ptrFromInt(x.ptr), 1);
    }

    /// SDOT: Dot product (float).
    pub fn sdot(self: Self, n: i32, x: driver.CudaSlice(f32), y: driver.CudaSlice(f32)) CublasError!f32 {
        var res: f32 = 0;
        try result.sdot(self.handle, n, @ptrFromInt(x.ptr), 1, @ptrFromInt(y.ptr), 1, &res);
        return res;
    }

    /// DDOT: Dot product (double).
    pub fn ddot(self: Self, n: i32, x: driver.CudaSlice(f64), y: driver.CudaSlice(f64)) CublasError!f64 {
        var res: f64 = 0;
        try result.ddot(self.handle, n, @ptrFromInt(x.ptr), 1, @ptrFromInt(y.ptr), 1, &res);
        return res;
    }

    /// SCOPY: y = x (float copy).
    pub fn scopy(self: Self, n: i32, x: driver.CudaSlice(f32), y: driver.CudaSlice(f32)) CublasError!void {
        try result.scopy(self.handle, n, @ptrFromInt(x.ptr), 1, @ptrFromInt(y.ptr), 1);
    }

    /// DCOPY: y = x (double copy).
    pub fn dcopy(self: Self, n: i32, x: driver.CudaSlice(f64), y: driver.CudaSlice(f64)) CublasError!void {
        try result.dcopy(self.handle, n, @ptrFromInt(x.ptr), 1, @ptrFromInt(y.ptr), 1);
    }

    /// SSWAP: swap x and y (float).
    pub fn sswap(self: Self, n: i32, x: driver.CudaSlice(f32), y: driver.CudaSlice(f32)) CublasError!void {
        try result.sswap(self.handle, n, @ptrFromInt(x.ptr), 1, @ptrFromInt(y.ptr), 1);
    }

    /// DSWAP: swap x and y (double).
    pub fn dswap(self: Self, n: i32, x: driver.CudaSlice(f64), y: driver.CudaSlice(f64)) CublasError!void {
        try result.dswap(self.handle, n, @ptrFromInt(x.ptr), 1, @ptrFromInt(y.ptr), 1);
    }

    /// ISAMAX: index of max absolute value (float). Returns 1-based index.
    pub fn isamax(self: Self, n: i32, x: driver.CudaSlice(f32)) CublasError!i32 {
        var idx: i32 = 0;
        try result.isamax(self.handle, n, @ptrFromInt(x.ptr), 1, &idx);
        return idx;
    }

    /// IDAMAX: index of max absolute value (double). Returns 1-based index.
    pub fn idamax(self: Self, n: i32, x: driver.CudaSlice(f64)) CublasError!i32 {
        var idx: i32 = 0;
        try result.idamax(self.handle, n, @ptrFromInt(x.ptr), 1, &idx);
        return idx;
    }

    /// ISAMIN: index of min absolute value (float). Returns 1-based index.
    pub fn isamin(self: Self, n: i32, x: driver.CudaSlice(f32)) CublasError!i32 {
        var idx: i32 = 0;
        try result.isamin(self.handle, n, @ptrFromInt(x.ptr), 1, &idx);
        return idx;
    }

    /// IDAMIN: index of min absolute value (double). Returns 1-based index.
    pub fn idamin(self: Self, n: i32, x: driver.CudaSlice(f64)) CublasError!i32 {
        var idx: i32 = 0;
        try result.idamin(self.handle, n, @ptrFromInt(x.ptr), 1, &idx);
        return idx;
    }

    // --- Double Precision Level 2 ---

    /// DGEMV: y = alpha * op(A) * x + beta * y (double).
    pub fn dgemv(
        self: Self,
        trans: Operation,
        m: i32,
        n: i32,
        alpha: f64,
        a: driver.CudaSlice(f64),
        lda: i32,
        x: driver.CudaSlice(f64),
        beta: f64,
        y: driver.CudaSlice(f64),
    ) CublasError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.dgemv(
            self.handle,
            toOp(trans),
            m,
            n,
            &alpha_val,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(x.ptr),
            1,
            &beta_val,
            @ptrFromInt(y.ptr),
            1,
        );
    }

    // --- Batched GEMM ---

    /// SGEMM Strided Batched: batch of C = alpha * A * B + beta * C (float).
    pub fn sgemmStridedBatched(
        self: Self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: driver.CudaSlice(f32),
        lda: i32,
        stride_a: i64,
        b: driver.CudaSlice(f32),
        ldb: i32,
        stride_b: i64,
        beta: f32,
        c_out: driver.CudaSlice(f32),
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) CublasError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.sgemmStridedBatched(
            self.handle,
            toOp(transa),
            toOp(transb),
            m,
            n,
            k,
            &alpha_val,
            @ptrFromInt(a.ptr),
            lda,
            stride_a,
            @ptrFromInt(b.ptr),
            ldb,
            stride_b,
            &beta_val,
            @ptrFromInt(c_out.ptr),
            ldc,
            stride_c,
            batch_count,
        );
    }

    /// DGEMM Strided Batched: batch of C = alpha * A * B + beta * C (double).
    pub fn dgemmStridedBatched(
        self: Self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: driver.CudaSlice(f64),
        lda: i32,
        stride_a: i64,
        b: driver.CudaSlice(f64),
        ldb: i32,
        stride_b: i64,
        beta: f64,
        c_out: driver.CudaSlice(f64),
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) CublasError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.dgemmStridedBatched(
            self.handle,
            toOp(transa),
            toOp(transb),
            m,
            n,
            k,
            &alpha_val,
            @ptrFromInt(a.ptr),
            lda,
            stride_a,
            @ptrFromInt(b.ptr),
            ldb,
            stride_b,
            &beta_val,
            @ptrFromInt(c_out.ptr),
            ldc,
            stride_c,
            batch_count,
        );
    }

    // --- Extended GEMM (mixed precision) ---

    /// GemmEx: mixed-precision GEMM. Supports f32 compute with f16/bf16/f32/f64 data.
    pub fn gemmEx(
        self: Self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: anytype,
        a_type: DataType,
        lda: i32,
        b: anytype,
        b_type: DataType,
        ldb: i32,
        beta: f32,
        c_out: anytype,
        c_type: DataType,
        ldc: i32,
    ) CublasError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.gemmEx(
            self.handle,
            toOp(transa),
            toOp(transb),
            m,
            n,
            k,
            @ptrCast(&alpha_val),
            @ptrFromInt(a.ptr),
            a_type.toSys(),
            lda,
            @ptrFromInt(b.ptr),
            b_type.toSys(),
            ldb,
            @ptrCast(&beta_val),
            @ptrFromInt(c_out.ptr),
            c_type.toSys(),
            ldc,
            sys.CUBLAS_COMPUTE_32F,
            sys.CUBLAS_GEMM_DEFAULT,
        );
    }

    /// Single-precision batched GEMM with pointer arrays.
    /// `a_array`, `b_array`, `c_array` are device pointers to arrays of device pointers.
    pub fn sgemmBatched(
        self: Self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_array: *const [*c]const f32,
        lda: i32,
        b_array: *const [*c]const f32,
        ldb: i32,
        beta: f32,
        c_array: *const [*c]f32,
        ldc: i32,
        batch_count: i32,
    ) CublasError!void {
        try result.sgemmBatched(
            self.handle,
            toOp(transa),
            toOp(transb),
            m,
            n,
            k,
            &alpha,
            a_array,
            lda,
            b_array,
            ldb,
            &beta,
            c_array,
            ldc,
            batch_count,
        );
    }

    /// Double-precision batched GEMM with pointer arrays.
    pub fn dgemmBatched(
        self: Self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a_array: *const [*c]const f64,
        lda: i32,
        b_array: *const [*c]const f64,
        ldb: i32,
        beta: f64,
        c_array: *const [*c]f64,
        ldc: i32,
        batch_count: i32,
    ) CublasError!void {
        try result.dgemmBatched(
            self.handle,
            toOp(transa),
            toOp(transb),
            m,
            n,
            k,
            &alpha,
            a_array,
            lda,
            b_array,
            ldb,
            &beta,
            c_array,
            ldc,
            batch_count,
        );
    }

    pub const Side = enum(sys.cublasSideMode_t) {
        left = sys.CUBLAS_SIDE_LEFT,
        right = sys.CUBLAS_SIDE_RIGHT,

        pub fn toSys(self: Side) sys.cublasSideMode_t {
            return @intFromEnum(self);
        }
    };

    pub const FillMode = enum(sys.cublasFillMode_t) {
        upper = sys.CUBLAS_FILL_MODE_UPPER,
        lower = sys.CUBLAS_FILL_MODE_LOWER,

        pub fn toSys(self: FillMode) sys.cublasFillMode_t {
            return @intFromEnum(self);
        }
    };

    pub const DiagType = enum(sys.cublasDiagType_t) {
        unit = sys.CUBLAS_DIAG_UNIT,
        non_unit = sys.CUBLAS_DIAG_NON_UNIT,

        pub fn toSys(self: DiagType) sys.cublasDiagType_t {
            return @intFromEnum(self);
        }
    };

    // --- Triangular Solve (TRSM) ---

    /// Single-precision triangular solve: op(A) * X = alpha * B.
    pub fn strsm(
        self: Self,
        side: Side,
        uplo: FillMode,
        transa: Operation,
        diag: DiagType,
        m: i32,
        n: i32,
        alpha: f32,
        a: anytype,
        lda: i32,
        b: anytype,
        ldb: i32,
    ) CublasError!void {
        try result.strsm(
            self.handle,
            side.toSys(),
            uplo.toSys(),
            toOp(transa),
            diag.toSys(),
            m,
            n,
            &alpha,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(b.ptr),
            ldb,
        );
    }

    /// Double-precision triangular solve.
    pub fn dtrsm(
        self: Self,
        side: Side,
        uplo: FillMode,
        transa: Operation,
        diag: DiagType,
        m: i32,
        n: i32,
        alpha: f64,
        a: anytype,
        lda: i32,
        b: anytype,
        ldb: i32,
    ) CublasError!void {
        try result.dtrsm(
            self.handle,
            side.toSys(),
            uplo.toSys(),
            toOp(transa),
            diag.toSys(),
            m,
            n,
            &alpha,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(b.ptr),
            ldb,
        );
    }

    // --- Symmetric Multiply (SYMM) ---

    /// Single-precision symmetric matrix multiply: C = alpha * A * B + beta * C.
    pub fn ssymm(
        self: Self,
        side: Side,
        uplo: FillMode,
        m: i32,
        n: i32,
        alpha: f32,
        a: anytype,
        lda: i32,
        b: anytype,
        ldb: i32,
        beta: f32,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.ssymm(
            self.handle,
            side.toSys(),
            uplo.toSys(),
            m,
            n,
            &alpha,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(b.ptr),
            ldb,
            &beta,
            @ptrFromInt(c_out.ptr),
            ldc,
        );
    }

    /// Double-precision symmetric matrix multiply.
    pub fn dsymm(
        self: Self,
        side: Side,
        uplo: FillMode,
        m: i32,
        n: i32,
        alpha: f64,
        a: anytype,
        lda: i32,
        b: anytype,
        ldb: i32,
        beta: f64,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.dsymm(
            self.handle,
            side.toSys(),
            uplo.toSys(),
            m,
            n,
            &alpha,
            @ptrFromInt(a.ptr),
            lda,
            @ptrFromInt(b.ptr),
            ldb,
            &beta,
            @ptrFromInt(c_out.ptr),
            ldc,
        );
    }

    // --- Symmetric Rank-k Update (SYRK) ---

    /// Single-precision symmetric rank-k update: C = alpha * A * A^T + beta * C.
    pub fn ssyrk(
        self: Self,
        uplo: FillMode,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: f32,
        a: anytype,
        lda: i32,
        beta: f32,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.ssyrk(self.handle, uplo.toSys(), toOp(trans), n, k, &alpha, @ptrFromInt(a.ptr), lda, &beta, @ptrFromInt(c_out.ptr), ldc);
    }

    /// Double-precision symmetric rank-k update.
    pub fn dsyrk(
        self: Self,
        uplo: FillMode,
        trans: Operation,
        n: i32,
        k: i32,
        alpha: f64,
        a: anytype,
        lda: i32,
        beta: f64,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.dsyrk(self.handle, uplo.toSys(), toOp(trans), n, k, &alpha, @ptrFromInt(a.ptr), lda, &beta, @ptrFromInt(c_out.ptr), ldc);
    }

    // --- Triangular Matrix Multiply (TRMM) ---

    /// Single-precision triangular matrix multiply: C = alpha * op(A) * B.
    pub fn strmm(
        self: Self,
        side: Side,
        uplo: FillMode,
        transa: Operation,
        diag: DiagType,
        m: i32,
        n: i32,
        alpha: f32,
        a: anytype,
        lda: i32,
        b: anytype,
        ldb: i32,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.strmm(self.handle, side.toSys(), uplo.toSys(), toOp(transa), diag.toSys(), m, n, &alpha, @ptrFromInt(a.ptr), lda, @ptrFromInt(b.ptr), ldb, @ptrFromInt(c_out.ptr), ldc);
    }

    /// Double-precision triangular matrix multiply.
    pub fn dtrmm(
        self: Self,
        side: Side,
        uplo: FillMode,
        transa: Operation,
        diag: DiagType,
        m: i32,
        n: i32,
        alpha: f64,
        a: anytype,
        lda: i32,
        b: anytype,
        ldb: i32,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.dtrmm(self.handle, side.toSys(), uplo.toSys(), toOp(transa), diag.toSys(), m, n, &alpha, @ptrFromInt(a.ptr), lda, @ptrFromInt(b.ptr), ldb, @ptrFromInt(c_out.ptr), ldc);
    }

    // --- Matrix Add/Transpose (GEAM) ---

    /// Single-precision GEAM: C = alpha * op(A) + beta * op(B).
    pub fn sgeam(
        self: Self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        alpha: f32,
        a: anytype,
        lda: i32,
        beta: f32,
        b: anytype,
        ldb: i32,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.sgeam(self.handle, toOp(transa), toOp(transb), m, n, &alpha, @ptrFromInt(a.ptr), lda, &beta, @ptrFromInt(b.ptr), ldb, @ptrFromInt(c_out.ptr), ldc);
    }

    /// Double-precision GEAM: C = alpha * op(A) + beta * op(B).
    pub fn dgeam(
        self: Self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        alpha: f64,
        a: anytype,
        lda: i32,
        beta: f64,
        b: anytype,
        ldb: i32,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.dgeam(self.handle, toOp(transa), toOp(transb), m, n, &alpha, @ptrFromInt(a.ptr), lda, &beta, @ptrFromInt(b.ptr), ldb, @ptrFromInt(c_out.ptr), ldc);
    }

    // --- Diagonal Matrix Multiply (DGMM) ---

    /// Single-precision DGMM: C = A * diag(x) or diag(x) * A.
    pub fn sdgmm(
        self: Self,
        side: Side,
        m: i32,
        n: i32,
        a: anytype,
        lda: i32,
        x: anytype,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.sdgmm(self.handle, side.toSys(), m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(x.ptr), 1, @ptrFromInt(c_out.ptr), ldc);
    }

    /// Double-precision DGMM: C = A * diag(x) or diag(x) * A.
    pub fn ddgmm(
        self: Self,
        side: Side,
        m: i32,
        n: i32,
        a: anytype,
        lda: i32,
        x: anytype,
        c_out: anytype,
        ldc: i32,
    ) CublasError!void {
        try result.ddgmm(self.handle, side.toSys(), m, n, @ptrFromInt(a.ptr), lda, @ptrFromInt(x.ptr), 1, @ptrFromInt(c_out.ptr), ldc);
    }

    // --- Grouped Batched GEMM (GemmGroupedBatchedEx) ---

    /// Grouped batched mixed-precision GEMM.
    /// Each group can have different m/n/k/transa/transb/lda/ldb/ldc.
    /// `group_count` groups, with `group_size[i]` batches in group i.
    pub fn gemmGroupedBatchedEx(
        self: Self,
        transa_array: []const sys.cublasOperation_t,
        transb_array: []const sys.cublasOperation_t,
        m_array: []const i32,
        n_array: []const i32,
        k_array: []const i32,
        alpha_array: *const anyopaque,
        a_array: [*c]const *const anyopaque,
        a_type: DataType,
        lda_array: []const i32,
        b_array: [*c]const *const anyopaque,
        b_type: DataType,
        ldb_array: []const i32,
        beta_array: *const anyopaque,
        c_array: [*c]const *anyopaque,
        c_type: DataType,
        ldc_array: []const i32,
        group_count: i32,
        group_size: []const i32,
        compute_type: sys.cublasComputeType_t,
    ) CublasError!void {
        try result.gemmGroupedBatchedEx(
            self.handle,
            transa_array.ptr,
            transb_array.ptr,
            m_array.ptr,
            n_array.ptr,
            k_array.ptr,
            alpha_array,
            a_array,
            a_type.toSys(),
            lda_array.ptr,
            b_array,
            b_type.toSys(),
            ldb_array.ptr,
            beta_array,
            c_array,
            c_type.toSys(),
            ldc_array.ptr,
            group_count,
            group_size.ptr,
            compute_type,
        );
    }
};

// ============================================================================
// Tests
// ============================================================================
