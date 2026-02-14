/// zCUDA: cuSPARSE - Safe abstraction layer.
///
/// Layer 3: High-level wrappers for cuSPARSE sparse operations (SpMV, SpMM).
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");
const driver = @import("../driver/driver.zig");

pub const CusparseError = result.CusparseError;

/// Sparse matrix format.
pub const SparseFormat = enum { csr, coo };

/// Sparse operation type.
pub const Operation = enum {
    non_transpose,
    transpose,
    conjugate_transpose,

    fn toSys(self: Operation) sys.cusparseOperation_t {
        return switch (self) {
            .non_transpose => sys.CUSPARSE_OPERATION_NON_TRANSPOSE,
            .transpose => sys.CUSPARSE_OPERATION_TRANSPOSE,
            .conjugate_transpose => sys.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
        };
    }
};

/// A cuSPARSE context.
pub const CusparseContext = struct {
    handle: sys.cusparseHandle_t,
    cuda_ctx: *const driver.CudaContext,

    const Self = @This();

    pub fn init(cuda_ctx: *const driver.CudaContext) !Self {
        try cuda_ctx.bindToThread();
        const handle = try result.create();
        return Self{ .handle = handle, .cuda_ctx = cuda_ctx };
    }

    pub fn deinit(self: Self) void {
        result.destroy(self.handle) catch {};
    }

    /// Set the CUDA stream for this handle.
    pub fn setStream(self: Self, stream: *const driver.CudaStream) CusparseError!void {
        try result.setStream(self.handle, stream.stream);
    }

    /// Create a CSR sparse matrix descriptor.
    pub fn createCsr(
        self: Self,
        rows: i64,
        cols: i64,
        nnz: i64,
        row_offsets: driver.CudaSlice(i32),
        col_indices: driver.CudaSlice(i32),
        values: driver.CudaSlice(f32),
    ) CusparseError!sys.cusparseSpMatDescr_t {
        _ = self;
        return result.createCsr(
            rows,
            cols,
            nnz,
            @ptrFromInt(row_offsets.ptr),
            @ptrFromInt(col_indices.ptr),
            @ptrFromInt(values.ptr),
            sys.CUSPARSE_INDEX_32I,
            sys.CUSPARSE_INDEX_32I,
            sys.CUSPARSE_INDEX_BASE_ZERO,
            sys.CUDA_R_32F,
        );
    }

    /// Create a dense vector descriptor.
    pub fn createDnVec(self: Self, data: driver.CudaSlice(f32)) CusparseError!sys.cusparseDnVecDescr_t {
        _ = self;
        return result.createDnVec(@intCast(data.len), @ptrFromInt(data.ptr), sys.CUDA_R_32F);
    }

    /// Create a dense matrix descriptor.
    pub fn createDnMat(
        self: Self,
        rows: i64,
        cols: i64,
        ld: i64,
        data: driver.CudaSlice(f32),
    ) CusparseError!sys.cusparseDnMatDescr_t {
        _ = self;
        return result.createDnMat(rows, cols, ld, @ptrFromInt(data.ptr), sys.CUDA_R_32F, sys.CUSPARSE_ORDER_COL);
    }

    /// Destroy a sparse matrix descriptor.
    pub fn destroySpMat(_: Self, mat: sys.cusparseSpMatDescr_t) void {
        result.destroySpMat(mat) catch {};
    }

    /// Destroy a dense vector descriptor.
    pub fn destroyDnVec(_: Self, vec: sys.cusparseDnVecDescr_t) void {
        result.destroyDnVec(vec) catch {};
    }

    /// Destroy a dense matrix descriptor.
    pub fn destroyDnMat(_: Self, mat: sys.cusparseDnMatDescr_t) void {
        result.destroyDnMat(mat) catch {};
    }

    /// SpMV: y = alpha * op(A) * x + beta * y (sparse A, dense x and y).
    pub fn spMV(
        self: Self,
        op: Operation,
        alpha: f32,
        mat_a: sys.cusparseSpMatDescr_t,
        vec_x: sys.cusparseDnVecDescr_t,
        beta: f32,
        vec_y: sys.cusparseDnVecDescr_t,
        workspace: ?driver.CudaSlice(u8),
    ) CusparseError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        const buf: ?*anyopaque = if (workspace) |ws| @ptrFromInt(ws.ptr) else null;
        try result.spMV(
            self.handle,
            op.toSys(),
            @ptrCast(&alpha_val),
            mat_a,
            vec_x,
            @ptrCast(&beta_val),
            vec_y,
            sys.CUDA_R_32F,
            sys.CUSPARSE_SPMV_ALG_DEFAULT,
            buf,
        );
    }

    /// Get workspace size for SpMV.
    pub fn spMV_bufferSize(
        self: Self,
        op: Operation,
        alpha: f32,
        mat_a: sys.cusparseSpMatDescr_t,
        vec_x: sys.cusparseDnVecDescr_t,
        beta: f32,
        vec_y: sys.cusparseDnVecDescr_t,
    ) CusparseError!usize {
        const alpha_val = alpha;
        const beta_val = beta;
        return result.spMV_bufferSize(
            self.handle,
            op.toSys(),
            @ptrCast(&alpha_val),
            mat_a,
            vec_x,
            @ptrCast(&beta_val),
            vec_y,
            sys.CUDA_R_32F,
            sys.CUSPARSE_SPMV_ALG_DEFAULT,
        );
    }

    /// SpMM: C = alpha * op(A) * op(B) + beta * C (sparse A, dense B and C).
    pub fn spMM(
        self: Self,
        op_a: Operation,
        op_b: Operation,
        alpha: f32,
        mat_a: sys.cusparseSpMatDescr_t,
        mat_b: sys.cusparseDnMatDescr_t,
        beta: f32,
        mat_c: sys.cusparseDnMatDescr_t,
        workspace: ?driver.CudaSlice(u8),
    ) CusparseError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        const buf: ?*anyopaque = if (workspace) |ws| @ptrFromInt(ws.ptr) else null;
        try result.spMM(
            self.handle,
            op_a.toSys(),
            op_b.toSys(),
            @ptrCast(&alpha_val),
            mat_a,
            mat_b,
            @ptrCast(&beta_val),
            mat_c,
            sys.CUDA_R_32F,
            sys.CUSPARSE_SPMM_ALG_DEFAULT,
            buf,
        );
    }

    /// Create a COO sparse matrix descriptor.
    pub fn createCoo(
        self: Self,
        rows: i64,
        cols: i64,
        nnz: i64,
        row_indices: driver.CudaSlice(i32),
        col_indices: driver.CudaSlice(i32),
        values: driver.CudaSlice(f32),
    ) CusparseError!sys.cusparseSpMatDescr_t {
        _ = self;
        return result.createCoo(
            rows,
            cols,
            nnz,
            @ptrFromInt(row_indices.ptr),
            @ptrFromInt(col_indices.ptr),
            @ptrFromInt(values.ptr),
            sys.CUSPARSE_INDEX_32I,
            sys.CUSPARSE_INDEX_BASE_ZERO,
            sys.CUDA_R_32F,
        );
    }

    /// Get buffer size for SpMM operation.
    pub fn spMMBufferSize(
        self: Self,
        op_a: Operation,
        op_b: Operation,
        alpha: f32,
        mat_a: sys.cusparseSpMatDescr_t,
        mat_b: sys.cusparseDnMatDescr_t,
        beta: f32,
        mat_c: sys.cusparseDnMatDescr_t,
    ) CusparseError!usize {
        const alpha_val = alpha;
        const beta_val = beta;
        return result.spMM_bufferSize(
            self.handle,
            op_a.toSys(),
            op_b.toSys(),
            @ptrCast(&alpha_val),
            mat_a,
            mat_b,
            @ptrCast(&beta_val),
            mat_c,
            sys.CUDA_R_32F,
            sys.CUSPARSE_SPMM_ALG_DEFAULT,
        );
    }

    /// Create a CSC (Compressed Sparse Column) sparse matrix descriptor.
    pub fn createCsc(
        self: Self,
        rows: i64,
        cols: i64,
        nnz: i64,
        col_offsets: driver.CudaSlice(i32),
        row_indices: driver.CudaSlice(i32),
        values: driver.CudaSlice(f32),
    ) CusparseError!sys.cusparseSpMatDescr_t {
        _ = self;
        return result.createCsc(
            rows,
            cols,
            nnz,
            @ptrFromInt(col_offsets.ptr),
            @ptrFromInt(row_indices.ptr),
            @ptrFromInt(values.ptr),
            sys.CUSPARSE_INDEX_32I,
            sys.CUSPARSE_INDEX_32I,
            sys.CUSPARSE_INDEX_BASE_ZERO,
            sys.CUDA_R_32F,
        );
    }

    // --- SpGEMM (Sparse × Sparse) ---

    /// Create an SpGEMM descriptor. Free with destroySpGEMMDescr().
    pub fn createSpGEMMDescr(self: Self) CusparseError!SpGEMMDescriptor {
        _ = self;
        return SpGEMMDescriptor.init();
    }

    /// SpGEMM work estimation phase.
    /// Call with null buffer to query size, then allocate and call again with buffer.
    pub fn spGEMM_workEstimation(
        self: Self,
        op_a: Operation,
        op_b: Operation,
        alpha: f32,
        mat_a: sys.cusparseSpMatDescr_t,
        mat_b: sys.cusparseSpMatDescr_t,
        beta: f32,
        mat_c: sys.cusparseSpMatDescr_t,
        alg: SpGEMMAlgorithm,
        spgemm_descr: SpGEMMDescriptor,
        buffer_size: *usize,
        buffer: ?*anyopaque,
    ) CusparseError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.spGEMM_workEstimation(
            self.handle,
            op_a.toSys(),
            op_b.toSys(),
            @ptrCast(&alpha_val),
            mat_a,
            mat_b,
            @ptrCast(&beta_val),
            mat_c,
            sys.CUDA_R_32F,
            alg.toSys(),
            spgemm_descr.descr,
            buffer_size,
            buffer,
        );
    }

    /// SpGEMM compute phase.
    /// Call with null buffer to query size, then allocate and call again with buffer.
    pub fn spGEMM_compute(
        self: Self,
        op_a: Operation,
        op_b: Operation,
        alpha: f32,
        mat_a: sys.cusparseSpMatDescr_t,
        mat_b: sys.cusparseSpMatDescr_t,
        beta: f32,
        mat_c: sys.cusparseSpMatDescr_t,
        alg: SpGEMMAlgorithm,
        spgemm_descr: SpGEMMDescriptor,
        buffer_size: *usize,
        buffer: ?*anyopaque,
    ) CusparseError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.spGEMM_compute(
            self.handle,
            op_a.toSys(),
            op_b.toSys(),
            @ptrCast(&alpha_val),
            mat_a,
            mat_b,
            @ptrCast(&beta_val),
            mat_c,
            sys.CUDA_R_32F,
            alg.toSys(),
            spgemm_descr.descr,
            buffer_size,
            buffer,
        );
    }

    /// SpGEMM copy phase — copy computed result into matC.
    pub fn spGEMM_copy(
        self: Self,
        op_a: Operation,
        op_b: Operation,
        alpha: f32,
        mat_a: sys.cusparseSpMatDescr_t,
        mat_b: sys.cusparseSpMatDescr_t,
        beta: f32,
        mat_c: sys.cusparseSpMatDescr_t,
        alg: SpGEMMAlgorithm,
        spgemm_descr: SpGEMMDescriptor,
    ) CusparseError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.spGEMM_copy(
            self.handle,
            op_a.toSys(),
            op_b.toSys(),
            @ptrCast(&alpha_val),
            mat_a,
            mat_b,
            @ptrCast(&beta_val),
            mat_c,
            sys.CUDA_R_32F,
            alg.toSys(),
            spgemm_descr.descr,
        );
    }
};

/// SpGEMM algorithm selection.
pub const SpGEMMAlgorithm = enum {
    default,
    csr_deterministic,
    csr_nondeterministic,

    fn toSys(self: SpGEMMAlgorithm) sys.cusparseSpGEMMAlg_t {
        return switch (self) {
            .default => sys.CUSPARSE_SPGEMM_DEFAULT,
            .csr_deterministic => sys.CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC,
            .csr_nondeterministic => sys.CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC,
        };
    }
};

/// An SpGEMM descriptor (RAII). Free with deinit().
pub const SpGEMMDescriptor = struct {
    descr: sys.cusparseSpGEMMDescr_t,

    pub fn init() CusparseError!SpGEMMDescriptor {
        const descr = try result.spGEMM_createDescr();
        return .{ .descr = descr };
    }

    pub fn deinit(self: SpGEMMDescriptor) void {
        result.spGEMM_destroyDescr(self.descr) catch {};
    }
};
