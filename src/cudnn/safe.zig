/// zCUDA: cuDNN API - Safe abstraction layer.
///
/// Layer 3: High-level safe wrappers for cuDNN deep learning operations.
/// Matches cudarc's cudnn/safe/ coverage: convolution, activation, pooling,
/// softmax, and reduction.
const std = @import("std");
const sys = @import("sys.zig");
const result = @import("result.zig");
const driver = @import("../driver/driver.zig");

pub const CudnnError = result.CudnnError;

// ============================================================================
// Enums
// ============================================================================

/// cuDNN data type.
pub const DnnDataType = enum {
    float,
    double,
    half,
    bfloat16,

    pub fn toSys(self: DnnDataType) sys.cudnnDataType_t {
        return switch (self) {
            .float => sys.CUDNN_DATA_FLOAT,
            .double => sys.CUDNN_DATA_DOUBLE,
            .half => sys.CUDNN_DATA_HALF,
            .bfloat16 => sys.CUDNN_DATA_BFLOAT16,
        };
    }
};

/// cuDNN tensor format.
pub const TensorFormat = enum {
    nchw,
    nhwc,

    pub fn toSys(self: TensorFormat) sys.cudnnTensorFormat_t {
        return switch (self) {
            .nchw => sys.CUDNN_TENSOR_NCHW,
            .nhwc => sys.CUDNN_TENSOR_NHWC,
        };
    }
};

/// Activation mode.
pub const ActivationMode = enum {
    sigmoid,
    relu,
    tanh,
    clipped_relu,
    elu,
    identity,

    fn toSys(self: ActivationMode) sys.cudnnActivationMode_t {
        return switch (self) {
            .sigmoid => sys.CUDNN_ACTIVATION_SIGMOID,
            .relu => sys.CUDNN_ACTIVATION_RELU,
            .tanh => sys.CUDNN_ACTIVATION_TANH,
            .clipped_relu => sys.CUDNN_ACTIVATION_CLIPPED_RELU,
            .elu => sys.CUDNN_ACTIVATION_ELU,
            .identity => sys.CUDNN_ACTIVATION_IDENTITY,
        };
    }
};

/// Convolution mode.
/// Output dimensions from convolution.
pub const ConvOutputDim = struct {
    n: i32,
    c: i32,
    h: i32,
    w: i32,
};

pub const ConvMode = enum {
    convolution,
    cross_correlation,

    fn toSys(self: ConvMode) sys.cudnnConvolutionMode_t {
        return switch (self) {
            .convolution => sys.CUDNN_CONVOLUTION,
            .cross_correlation => sys.CUDNN_CROSS_CORRELATION,
        };
    }
};

/// Convolution forward algorithm.
pub const ConvFwdAlgo = enum {
    implicit_gemm,
    implicit_precomp_gemm,
    gemm,
    direct,
    fft,
    fft_tiling,
    winograd,
    winograd_nonfused,

    fn toSys(self: ConvFwdAlgo) sys.cudnnConvolutionFwdAlgo_t {
        return switch (self) {
            .implicit_gemm => sys.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
            .implicit_precomp_gemm => sys.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            .gemm => sys.CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            .direct => sys.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
            .fft => sys.CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            .fft_tiling => sys.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
            .winograd => sys.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            .winograd_nonfused => sys.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        };
    }
};

/// Pooling mode.
pub const PoolingMode = enum {
    max,
    avg_count_include_padding,
    avg_count_exclude_padding,
    max_deterministic,

    fn toSys(self: PoolingMode) sys.cudnnPoolingMode_t {
        return switch (self) {
            .max => sys.CUDNN_POOLING_MAX,
            .avg_count_include_padding => sys.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
            .avg_count_exclude_padding => sys.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
            .max_deterministic => sys.CUDNN_POOLING_MAX_DETERMINISTIC,
        };
    }
};

/// Softmax algorithm.
pub const SoftmaxAlgo = enum {
    fast,
    accurate,
    log,

    fn toSys(self: SoftmaxAlgo) sys.cudnnSoftmaxAlgorithm_t {
        return switch (self) {
            .fast => sys.CUDNN_SOFTMAX_FAST,
            .accurate => sys.CUDNN_SOFTMAX_ACCURATE,
            .log => sys.CUDNN_SOFTMAX_LOG,
        };
    }
};

/// Softmax mode.
pub const SoftmaxMode = enum {
    instance,
    channel,

    fn toSys(self: SoftmaxMode) sys.cudnnSoftmaxMode_t {
        return switch (self) {
            .instance => sys.CUDNN_SOFTMAX_MODE_INSTANCE,
            .channel => sys.CUDNN_SOFTMAX_MODE_CHANNEL,
        };
    }
};

/// Reduce tensor operation.
pub const ReduceOp = enum {
    add,
    mul,
    min,
    max,
    amax,
    avg,
    norm1,
    norm2,

    fn toSys(self: ReduceOp) sys.cudnnReduceTensorOp_t {
        return switch (self) {
            .add => sys.CUDNN_REDUCE_TENSOR_ADD,
            .mul => sys.CUDNN_REDUCE_TENSOR_MUL,
            .min => sys.CUDNN_REDUCE_TENSOR_MIN,
            .max => sys.CUDNN_REDUCE_TENSOR_MAX,
            .amax => sys.CUDNN_REDUCE_TENSOR_AMAX,
            .avg => sys.CUDNN_REDUCE_TENSOR_AVG,
            .norm1 => sys.CUDNN_REDUCE_TENSOR_NORM1,
            .norm2 => sys.CUDNN_REDUCE_TENSOR_NORM2,
        };
    }
};

/// Element-wise tensor operation.
pub const OpTensorOp = enum {
    add,
    mul,
    min,
    max,
    sqrt,
    not,

    fn toSys(self: OpTensorOp) sys.cudnnOpTensorOp_t {
        return switch (self) {
            .add => sys.CUDNN_OP_TENSOR_ADD,
            .mul => sys.CUDNN_OP_TENSOR_MUL,
            .min => sys.CUDNN_OP_TENSOR_MIN,
            .max => sys.CUDNN_OP_TENSOR_MAX,
            .sqrt => sys.CUDNN_OP_TENSOR_SQRT,
            .not => sys.CUDNN_OP_TENSOR_NOT,
        };
    }
};

/// Batch normalization mode.
pub const BatchNormMode = enum {
    per_activation,
    spatial,
    spatial_persistent,

    pub fn toSys(self: BatchNormMode) sys.cudnnBatchNormMode_t {
        return switch (self) {
            .per_activation => sys.CUDNN_BATCHNORM_PER_ACTIVATION,
            .spatial => sys.CUDNN_BATCHNORM_SPATIAL,
            .spatial_persistent => sys.CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        };
    }
};

// ============================================================================
// Descriptor Types (RAII wrappers)
// ============================================================================

/// A dropout descriptor. Free with deinit().
pub const DropoutDescriptor = struct {
    desc: sys.cudnnDropoutDescriptor_t,

    pub fn deinit(self: DropoutDescriptor) void {
        result.destroyDropoutDescriptor(self.desc) catch {};
    }
};

/// A 4D tensor descriptor. Free with deinit().
pub const TensorDescriptor = struct {
    desc: sys.cudnnTensorDescriptor_t,

    pub fn init(format: TensorFormat, data_type: DnnDataType, n: i32, c: i32, h: i32, w: i32) CudnnError!TensorDescriptor {
        const desc = try result.createTensorDescriptor();
        try result.setTensor4dDescriptor(desc, format.toSys(), data_type.toSys(), n, c, h, w);
        return .{ .desc = desc };
    }

    pub fn deinit(self: TensorDescriptor) void {
        result.destroyTensorDescriptor(self.desc) catch {};
    }
};

/// A filter descriptor. Free with deinit().
pub const FilterDescriptor = struct {
    desc: sys.cudnnFilterDescriptor_t,

    pub fn init(data_type: DnnDataType, format: TensorFormat, k: i32, c: i32, h: i32, w: i32) CudnnError!FilterDescriptor {
        const desc = try result.createFilterDescriptor();
        try result.setFilter4dDescriptor(desc, data_type.toSys(), format.toSys(), k, c, h, w);
        return .{ .desc = desc };
    }

    pub fn deinit(self: FilterDescriptor) void {
        result.destroyFilterDescriptor(self.desc) catch {};
    }
};

/// A convolution descriptor. Free with deinit().
pub const ConvolutionDescriptor = struct {
    desc: sys.cudnnConvolutionDescriptor_t,

    pub fn init2d(
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        mode: ConvMode,
        data_type: DnnDataType,
    ) CudnnError!ConvolutionDescriptor {
        const desc = try result.createConvolutionDescriptor();
        try result.setConvolution2dDescriptor(
            desc,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            mode.toSys(),
            data_type.toSys(),
        );
        return .{ .desc = desc };
    }

    pub fn deinit(self: ConvolutionDescriptor) void {
        result.destroyConvolutionDescriptor(self.desc) catch {};
    }
};

/// An N-dimensional tensor descriptor (5D for 3D conv). Free with deinit().
pub const TensorDescriptorNd = struct {
    desc: sys.cudnnTensorDescriptor_t,

    /// Create an Nd tensor descriptor. `dims` and `strides` must have the same length.
    pub fn init(data_type: DnnDataType, dims: []const i32, strides: []const i32) CudnnError!TensorDescriptorNd {
        const desc = try result.createTensorDescriptor();
        try result.setTensorNdDescriptor(desc, data_type.toSys(), @intCast(dims.len), dims.ptr, strides.ptr);
        return .{ .desc = desc };
    }

    pub fn deinit(self: TensorDescriptorNd) void {
        result.destroyTensorDescriptor(self.desc) catch {};
    }
};

/// An N-dimensional filter descriptor (5D for 3D conv). Free with deinit().
pub const FilterDescriptorNd = struct {
    desc: sys.cudnnFilterDescriptor_t,

    /// Create an Nd filter descriptor. `filter_dims` length determines dimensionality.
    pub fn init(data_type: DnnDataType, format: TensorFormat, filter_dims: []const i32) CudnnError!FilterDescriptorNd {
        const desc = try result.createFilterDescriptor();
        try result.setFilterNdDescriptor(desc, data_type.toSys(), format.toSys(), @intCast(filter_dims.len), filter_dims.ptr);
        return .{ .desc = desc };
    }

    pub fn deinit(self: FilterDescriptorNd) void {
        result.destroyFilterDescriptor(self.desc) catch {};
    }
};

/// An N-dimensional convolution descriptor (3D conv). Free with deinit().
pub const ConvolutionDescriptorNd = struct {
    desc: sys.cudnnConvolutionDescriptor_t,

    /// Create an Nd convolution descriptor.
    /// `pads`, `strides`, `dilations` must all have `spatial_dims` elements (e.g., 3 for 3D conv).
    pub fn init(
        pads: []const i32,
        strides: []const i32,
        dilations: []const i32,
        mode: ConvMode,
        data_type: DnnDataType,
    ) CudnnError!ConvolutionDescriptorNd {
        const desc = try result.createConvolutionDescriptor();
        try result.setConvolutionNdDescriptor(
            desc,
            @intCast(pads.len),
            pads.ptr,
            strides.ptr,
            dilations.ptr,
            mode.toSys(),
            data_type.toSys(),
        );
        return .{ .desc = desc };
    }

    pub fn deinit(self: ConvolutionDescriptorNd) void {
        result.destroyConvolutionDescriptor(self.desc) catch {};
    }
};

/// An activation descriptor. Free with deinit().
pub const ActivationDescriptor = struct {
    desc: sys.cudnnActivationDescriptor_t,

    pub fn init(mode: ActivationMode, coef: f64) CudnnError!ActivationDescriptor {
        const desc = try result.createActivationDescriptor();
        try result.setActivationDescriptor(desc, mode.toSys(), sys.CUDNN_NOT_PROPAGATE_NAN, coef);
        return .{ .desc = desc };
    }

    pub fn deinit(self: ActivationDescriptor) void {
        result.destroyActivationDescriptor(self.desc) catch {};
    }
};

/// A pooling descriptor. Free with deinit().
pub const PoolingDescriptor = struct {
    desc: sys.cudnnPoolingDescriptor_t,

    pub fn init2d(
        mode: PoolingMode,
        window_h: i32,
        window_w: i32,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
    ) CudnnError!PoolingDescriptor {
        const desc = try result.createPoolingDescriptor();
        try result.setPooling2dDescriptor(
            desc,
            mode.toSys(),
            sys.CUDNN_NOT_PROPAGATE_NAN,
            window_h,
            window_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
        );
        return .{ .desc = desc };
    }

    pub fn deinit(self: PoolingDescriptor) void {
        result.destroyPoolingDescriptor(self.desc) catch {};
    }
};

/// An op tensor descriptor. Free with deinit().
pub const OpTensorDescriptor = struct {
    desc: sys.cudnnOpTensorDescriptor_t,

    pub fn init(op: OpTensorOp, comp_type: DnnDataType) CudnnError!OpTensorDescriptor {
        const desc = try result.createOpTensorDescriptor();
        try result.setOpTensorDescriptor(desc, op.toSys(), comp_type.toSys(), sys.CUDNN_NOT_PROPAGATE_NAN);
        return .{ .desc = desc };
    }

    pub fn deinit(self: OpTensorDescriptor) void {
        result.destroyOpTensorDescriptor(self.desc) catch {};
    }
};

// ============================================================================
// CudnnContext — Main cuDNN handle
// ============================================================================

/// A cuDNN context wrapping a cuDNN handle.
pub const CudnnContext = struct {
    handle: sys.cudnnHandle_t,
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

    pub fn version() usize {
        return result.getVersion();
    }

    /// Set the CUDA stream for this handle.
    pub fn setStream(self: Self, stream: *const driver.CudaStream) CudnnError!void {
        try result.setStream(self.handle, stream.stream);
    }

    // --- Tensor Descriptors ---

    /// Create a 4D tensor descriptor.
    pub fn createTensor4d(
        self: Self,
        format: TensorFormat,
        data_type: DnnDataType,
        n: i32,
        channels: i32,
        h: i32,
        w: i32,
    ) CudnnError!TensorDescriptor {
        _ = self;
        return TensorDescriptor.init(format, data_type, n, channels, h, w);
    }

    /// Create a 4D filter descriptor.
    pub fn createFilter4d(
        self: Self,
        data_type: DnnDataType,
        format: TensorFormat,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
    ) CudnnError!FilterDescriptor {
        _ = self;
        return FilterDescriptor.init(data_type, format, k, c, h, w);
    }

    // --- Convolution ---

    /// Create a 2D convolution descriptor.
    pub fn createConv2d(
        self: Self,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        mode: ConvMode,
        data_type: DnnDataType,
    ) CudnnError!ConvolutionDescriptor {
        _ = self;
        return ConvolutionDescriptor.init2d(pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, mode, data_type);
    }

    /// Get the output dimensions for a convolution forward.
    pub fn getConvOutputDim(
        self: Self,
        conv_desc: ConvolutionDescriptor,
        input_desc: TensorDescriptor,
        filter_desc: FilterDescriptor,
    ) CudnnError!ConvOutputDim {
        _ = self;
        const r = try result.getConvolution2dForwardOutputDim(conv_desc.desc, input_desc.desc, filter_desc.desc);
        return .{ .n = r.n, .c = r.c, .h = r.h, .w = r.w };
    }

    /// Get workspace size for convolution forward.
    pub fn convForwardWorkspaceSize(
        self: Self,
        x_desc: TensorDescriptor,
        w_desc: FilterDescriptor,
        conv_desc: ConvolutionDescriptor,
        y_desc: TensorDescriptor,
        algo: ConvFwdAlgo,
    ) CudnnError!usize {
        return result.getConvolutionForwardWorkspaceSize(
            self.handle,
            x_desc.desc,
            w_desc.desc,
            conv_desc.desc,
            y_desc.desc,
            algo.toSys(),
        );
    }

    /// Convolution forward: y = conv(x, w).
    pub fn convForward(
        self: Self,
        comptime T: type,
        alpha: T,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        w_desc: FilterDescriptor,
        w: driver.CudaSlice(T),
        conv_desc: ConvolutionDescriptor,
        algo: ConvFwdAlgo,
        workspace: ?driver.CudaSlice(u8),
        beta: T,
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        const ws_ptr: ?*anyopaque = if (workspace) |ws| @ptrFromInt(ws.ptr) else null;
        const ws_size: usize = if (workspace) |ws| ws.len else 0;
        try result.convolutionForward(
            self.handle,
            @ptrCast(&alpha_val),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            w_desc.desc,
            @ptrFromInt(w.ptr),
            conv_desc.desc,
            algo.toSys(),
            ws_ptr,
            ws_size,
            @ptrCast(&beta_val),
            y_desc.desc,
            @ptrFromInt(y.ptr),
        );
    }

    // --- N-dimensional Convolution (3D+) ---

    /// Create an N-dimensional tensor descriptor.
    pub fn createTensorNd(self: Self, data_type: DnnDataType, dims: []const i32, strides: []const i32) CudnnError!TensorDescriptorNd {
        _ = self;
        return TensorDescriptorNd.init(data_type, dims, strides);
    }

    /// Create an N-dimensional filter descriptor.
    pub fn createFilterNd(self: Self, data_type: DnnDataType, format: TensorFormat, filter_dims: []const i32) CudnnError!FilterDescriptorNd {
        _ = self;
        return FilterDescriptorNd.init(data_type, format, filter_dims);
    }

    /// Create an N-dimensional convolution descriptor.
    pub fn createConvNd(
        self: Self,
        pads: []const i32,
        strides: []const i32,
        dilations: []const i32,
        mode: ConvMode,
        data_type: DnnDataType,
    ) CudnnError!ConvolutionDescriptorNd {
        _ = self;
        return ConvolutionDescriptorNd.init(pads, strides, dilations, mode, data_type);
    }

    /// Create a 3D convolution descriptor (convenience).
    pub fn createConv3d(
        self: Self,
        pad: [3]i32,
        stride: [3]i32,
        dilation: [3]i32,
        mode: ConvMode,
        data_type: DnnDataType,
    ) CudnnError!ConvolutionDescriptorNd {
        _ = self;
        return ConvolutionDescriptorNd.init(&pad, &stride, &dilation, mode, data_type);
    }

    /// Get output dimensions for N-dimensional convolution forward.
    pub fn getConvNdForwardOutputDim(
        self: Self,
        conv_desc: ConvolutionDescriptorNd,
        input_desc: TensorDescriptorNd,
        filter_desc: FilterDescriptorNd,
        output_dims: []i32,
    ) CudnnError!void {
        _ = self;
        try result.getConvolutionNdForwardOutputDim(
            conv_desc.desc,
            input_desc.desc,
            filter_desc.desc,
            @intCast(output_dims.len),
            output_dims.ptr,
        );
    }

    /// Convolution forward with N-dimensional descriptors.
    pub fn convForwardNd(
        self: Self,
        comptime T: type,
        alpha: T,
        x_desc: TensorDescriptorNd,
        x: driver.CudaSlice(T),
        w_desc: FilterDescriptorNd,
        w: driver.CudaSlice(T),
        conv_desc: ConvolutionDescriptorNd,
        algo: ConvFwdAlgo,
        workspace: ?driver.CudaSlice(u8),
        beta: T,
        y_desc: TensorDescriptorNd,
        y: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        const ws_ptr: ?*anyopaque = if (workspace) |ws| @ptrFromInt(ws.ptr) else null;
        const ws_size: usize = if (workspace) |ws| ws.len else 0;
        try result.convolutionForward(
            self.handle,
            @ptrCast(&alpha_val),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            w_desc.desc,
            @ptrFromInt(w.ptr),
            conv_desc.desc,
            algo.toSys(),
            ws_ptr,
            ws_size,
            @ptrCast(&beta_val),
            y_desc.desc,
            @ptrFromInt(y.ptr),
        );
    }

    // --- Element-wise Tensor Operations ---

    /// Element-wise tensor operation: C = op(alpha1*A, alpha2*B) + beta*C.
    pub fn opTensor(
        self: Self,
        comptime T: type,
        op_desc: OpTensorDescriptor,
        alpha1: T,
        a_desc: TensorDescriptor,
        a: driver.CudaSlice(T),
        alpha2: T,
        b_desc: TensorDescriptor,
        b: driver.CudaSlice(T),
        beta: T,
        c_desc: TensorDescriptor,
        c: driver.CudaSlice(T),
    ) CudnnError!void {
        const a1 = alpha1;
        const a2 = alpha2;
        const beta_val = beta;
        try result.opTensor(
            self.handle,
            op_desc.desc,
            @ptrCast(&a1),
            a_desc.desc,
            @ptrFromInt(a.ptr),
            @ptrCast(&a2),
            b_desc.desc,
            @ptrFromInt(b.ptr),
            @ptrCast(&beta_val),
            c_desc.desc,
            @ptrFromInt(c.ptr),
        );
    }

    /// Add a tensor: C = alpha * A + beta * C.
    pub fn addTensor(
        self: Self,
        comptime T: type,
        alpha: T,
        a_desc: TensorDescriptor,
        a: driver.CudaSlice(T),
        beta: T,
        c_desc: TensorDescriptor,
        c: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.addTensor(
            self.handle,
            @ptrCast(&alpha_val),
            a_desc.desc,
            @ptrFromInt(a.ptr),
            @ptrCast(&beta_val),
            c_desc.desc,
            @ptrFromInt(c.ptr),
        );
    }

    /// Scale a tensor in-place: Y = alpha * Y.
    pub fn scaleTensor(
        self: Self,
        comptime T: type,
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
        alpha: T,
    ) CudnnError!void {
        const alpha_val = alpha;
        try result.scaleTensor(
            self.handle,
            y_desc.desc,
            @ptrFromInt(y.ptr),
            @ptrCast(&alpha_val),
        );
    }

    /// Get workspace size for reduce tensor operation.
    pub fn getReductionWorkspaceSize(
        self: Self,
        reduce_op: ReduceOp,
        a_desc: TensorDescriptor,
        c_desc: TensorDescriptor,
    ) CudnnError!usize {
        const reduce_desc = try result.createReduceTensorDescriptor();
        defer result.destroyReduceTensorDescriptor(reduce_desc) catch {};
        try result.setReduceTensorDescriptor(
            reduce_desc,
            reduce_op.toSys(),
            DnnDataType.float.toSys(),
            sys.CUDNN_NOT_PROPAGATE_NAN,
            sys.CUDNN_REDUCE_TENSOR_NO_INDICES,
            sys.CUDNN_32BIT_INDICES,
        );
        return result.getReductionWorkspaceSize(self.handle, reduce_desc, a_desc.desc, c_desc.desc);
    }

    // --- Activation ---

    /// Activation forward: y = activation(x).
    pub fn activationForward(
        self: Self,
        comptime T: type,
        act_desc: ActivationDescriptor,
        alpha: T,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        beta: T,
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.activationForward(
            self.handle,
            act_desc.desc,
            @ptrCast(&alpha_val),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            @ptrCast(&beta_val),
            y_desc.desc,
            @ptrFromInt(y.ptr),
        );
    }

    // --- Pooling ---

    /// Pooling forward.
    pub fn poolingForward(
        self: Self,
        comptime T: type,
        pool_desc: PoolingDescriptor,
        alpha: T,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        beta: T,
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.poolingForward(
            self.handle,
            pool_desc.desc,
            @ptrCast(&alpha_val),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            @ptrCast(&beta_val),
            y_desc.desc,
            @ptrFromInt(y.ptr),
        );
    }

    /// Pooling backward: compute dx from y, dy, x.
    pub fn poolingBackward(
        self: Self,
        comptime T: type,
        pool_desc: PoolingDescriptor,
        alpha: T,
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
        dy_desc: TensorDescriptor,
        dy: driver.CudaSlice(T),
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        beta: T,
        dx_desc: TensorDescriptor,
        dx: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.poolingBackward(
            self.handle,
            pool_desc.desc,
            @ptrCast(&alpha_val),
            y_desc.desc,
            @ptrFromInt(y.ptr),
            dy_desc.desc,
            @ptrFromInt(dy.ptr),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            @ptrCast(&beta_val),
            dx_desc.desc,
            @ptrFromInt(dx.ptr),
        );
    }

    // --- Softmax ---

    /// Softmax forward.
    pub fn softmaxForward(
        self: Self,
        comptime T: type,
        algo: SoftmaxAlgo,
        mode: SoftmaxMode,
        alpha: T,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        beta: T,
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.softmaxForward(
            self.handle,
            algo.toSys(),
            mode.toSys(),
            @ptrCast(&alpha_val),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            @ptrCast(&beta_val),
            y_desc.desc,
            @ptrFromInt(y.ptr),
        );
    }

    /// Softmax backward.
    pub fn softmaxBackward(
        self: Self,
        comptime T: type,
        algo: SoftmaxAlgo,
        mode: SoftmaxMode,
        alpha: T,
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
        dy_desc: TensorDescriptor,
        dy: driver.CudaSlice(T),
        beta: T,
        dx_desc: TensorDescriptor,
        dx: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.softmaxBackward(
            self.handle,
            algo.toSys(),
            mode.toSys(),
            @ptrCast(&alpha_val),
            y_desc.desc,
            @ptrFromInt(y.ptr),
            dy_desc.desc,
            @ptrFromInt(dy.ptr),
            @ptrCast(&beta_val),
            dx_desc.desc,
            @ptrFromInt(dx.ptr),
        );
    }

    // --- Reduction ---

    /// Reduce a tensor (e.g., sum, max, norm).
    pub fn reduceTensor(
        self: Self,
        comptime T: type,
        op: ReduceOp,
        alpha: T,
        a_desc: TensorDescriptor,
        a: driver.CudaSlice(T),
        beta: T,
        c_desc: TensorDescriptor,
        c: driver.CudaSlice(T),
        workspace: ?driver.CudaSlice(u8),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        const ws_ptr: ?*anyopaque = if (workspace) |ws| @ptrFromInt(ws.ptr) else null;
        const ws_size: usize = if (workspace) |ws| ws.len else 0;

        // Create reduce descriptor on the fly
        const reduce_desc = try result.createReduceTensorDescriptor();
        defer result.destroyReduceTensorDescriptor(reduce_desc) catch {};
        const comp_type = DnnDataType.float;
        try result.setReduceTensorDescriptor(
            reduce_desc,
            op.toSys(),
            comp_type.toSys(),
            sys.CUDNN_NOT_PROPAGATE_NAN,
            sys.CUDNN_REDUCE_TENSOR_NO_INDICES,
            sys.CUDNN_32BIT_INDICES,
        );

        try result.reduceTensor(
            self.handle,
            reduce_desc,
            null,
            0,
            ws_ptr,
            ws_size,
            @ptrCast(&alpha_val),
            a_desc.desc,
            @ptrFromInt(a.ptr),
            @ptrCast(&beta_val),
            c_desc.desc,
            @ptrFromInt(c.ptr),
        );
    }

    // --- Convolution Backward Data ---

    /// Convolution backward data: compute dx from dy and w.
    pub fn convBackwardData(
        self: Self,
        comptime T: type,
        alpha: T,
        w_desc: FilterDescriptor,
        w: driver.CudaSlice(T),
        dy_desc: TensorDescriptor,
        dy: driver.CudaSlice(T),
        conv_desc: ConvolutionDescriptor,
        workspace: ?driver.CudaSlice(u8),
        beta: T,
        dx_desc: TensorDescriptor,
        dx: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        const ws_ptr: ?*anyopaque = if (workspace) |ws| @ptrFromInt(ws.ptr) else null;
        const ws_size: usize = if (workspace) |ws| ws.len else 0;
        try result.convolutionBackwardData(
            self.handle,
            @ptrCast(&alpha_val),
            w_desc.desc,
            @ptrFromInt(w.ptr),
            dy_desc.desc,
            @ptrFromInt(dy.ptr),
            conv_desc.desc,
            sys.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
            ws_ptr,
            ws_size,
            @ptrCast(&beta_val),
            dx_desc.desc,
            @ptrFromInt(dx.ptr),
        );
    }

    // --- Convolution Backward Filter ---

    /// Convolution backward filter: compute dw from x and dy.
    pub fn convBackwardFilter(
        self: Self,
        comptime T: type,
        alpha: T,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        dy_desc: TensorDescriptor,
        dy: driver.CudaSlice(T),
        conv_desc: ConvolutionDescriptor,
        workspace: ?driver.CudaSlice(u8),
        beta: T,
        dw_desc: FilterDescriptor,
        dw: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        const ws_ptr: ?*anyopaque = if (workspace) |ws| @ptrFromInt(ws.ptr) else null;
        const ws_size: usize = if (workspace) |ws| ws.len else 0;
        try result.convolutionBackwardFilter(
            self.handle,
            @ptrCast(&alpha_val),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            dy_desc.desc,
            @ptrFromInt(dy.ptr),
            conv_desc.desc,
            sys.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
            ws_ptr,
            ws_size,
            @ptrCast(&beta_val),
            dw_desc.desc,
            @ptrFromInt(dw.ptr),
        );
    }

    // --- Fused Conv + Bias + Activation ---

    /// Fused convolution + bias + activation: y = activation(conv(x, w) + bias).
    pub fn convBiasActivationForward(
        self: Self,
        comptime T: type,
        alpha1: T,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        w_desc: FilterDescriptor,
        w: driver.CudaSlice(T),
        conv_desc: ConvolutionDescriptor,
        algo: ConvFwdAlgo,
        workspace: ?driver.CudaSlice(u8),
        alpha2: T,
        z_desc: TensorDescriptor,
        z: driver.CudaSlice(T),
        bias_desc: TensorDescriptor,
        bias: driver.CudaSlice(T),
        act_desc: ActivationDescriptor,
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
    ) CudnnError!void {
        const a1 = alpha1;
        const a2 = alpha2;
        const ws_ptr: ?*anyopaque = if (workspace) |ws| @ptrFromInt(ws.ptr) else null;
        const ws_size: usize = if (workspace) |ws| ws.len else 0;
        try result.convolutionBiasActivationForward(
            self.handle,
            @ptrCast(&a1),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            w_desc.desc,
            @ptrFromInt(w.ptr),
            conv_desc.desc,
            algo.toSys(),
            ws_ptr,
            ws_size,
            @ptrCast(&a2),
            z_desc.desc,
            @ptrFromInt(z.ptr),
            bias_desc.desc,
            @ptrFromInt(bias.ptr),
            act_desc.desc,
            y_desc.desc,
            @ptrFromInt(y.ptr),
        );
    }

    // --- Activation Backward ---

    /// Activation backward: compute dx from x, y, dy.
    pub fn activationBackward(
        self: Self,
        comptime T: type,
        act_desc: ActivationDescriptor,
        alpha: T,
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
        dy_desc: TensorDescriptor,
        dy: driver.CudaSlice(T),
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        beta: T,
        dx_desc: TensorDescriptor,
        dx: driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val = beta;
        try result.activationBackward(
            self.handle,
            act_desc.desc,
            @ptrCast(&alpha_val),
            y_desc.desc,
            @ptrFromInt(y.ptr),
            dy_desc.desc,
            @ptrFromInt(dy.ptr),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            @ptrCast(&beta_val),
            dx_desc.desc,
            @ptrFromInt(dx.ptr),
        );
    }

    // --- Batch Normalization ---

    /// Derive the BN parameter tensor descriptor from input tensor descriptor.
    pub fn deriveBNDescriptor(
        self: Self,
        x_desc: TensorDescriptor,
        mode: BatchNormMode,
    ) CudnnError!TensorDescriptor {
        _ = self;
        const derived = try result.createTensorDescriptor();
        try result.deriveBNTensorDescriptor(derived, x_desc.desc, mode.toSys());
        return .{ .desc = derived };
    }

    /// Batch normalization forward (training mode).
    pub fn batchNormForwardTraining(
        self: Self,
        comptime T: type,
        mode: BatchNormMode,
        alpha: T,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
        bn_desc: TensorDescriptor,
        scale: driver.CudaSlice(T),
        bias: driver.CudaSlice(T),
        exp_avg_factor: f64,
        running_mean: ?driver.CudaSlice(T),
        running_var: ?driver.CudaSlice(T),
        epsilon: f64,
        save_mean: ?driver.CudaSlice(T),
        save_inv_var: ?driver.CudaSlice(T),
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val: T = 0;
        try result.batchNormalizationForwardTraining(
            self.handle,
            mode.toSys(),
            @ptrCast(&alpha_val),
            @ptrCast(&beta_val),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            y_desc.desc,
            @ptrFromInt(y.ptr),
            bn_desc.desc,
            @ptrFromInt(scale.ptr),
            @ptrFromInt(bias.ptr),
            exp_avg_factor,
            if (running_mean) |m| @ptrFromInt(m.ptr) else null,
            if (running_var) |v| @ptrFromInt(v.ptr) else null,
            epsilon,
            if (save_mean) |m| @ptrFromInt(m.ptr) else null,
            if (save_inv_var) |v| @ptrFromInt(v.ptr) else null,
        );
    }

    /// Batch normalization forward (inference mode).
    pub fn batchNormForwardInference(
        self: Self,
        comptime T: type,
        mode: BatchNormMode,
        alpha: T,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
        bn_desc: TensorDescriptor,
        scale: driver.CudaSlice(T),
        bias: driver.CudaSlice(T),
        estimated_mean: driver.CudaSlice(T),
        estimated_var: driver.CudaSlice(T),
        epsilon: f64,
    ) CudnnError!void {
        const alpha_val = alpha;
        const beta_val: T = 0;
        try result.batchNormalizationForwardInference(
            self.handle,
            mode.toSys(),
            @ptrCast(&alpha_val),
            @ptrCast(&beta_val),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            y_desc.desc,
            @ptrFromInt(y.ptr),
            bn_desc.desc,
            @ptrFromInt(scale.ptr),
            @ptrFromInt(bias.ptr),
            @ptrFromInt(estimated_mean.ptr),
            @ptrFromInt(estimated_var.ptr),
            epsilon,
        );
    }

    /// Batch normalization backward.
    pub fn batchNormBackward(
        self: Self,
        comptime T: type,
        mode: BatchNormMode,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        dy_desc: TensorDescriptor,
        dy: driver.CudaSlice(T),
        dx_desc: TensorDescriptor,
        dx: driver.CudaSlice(T),
        bn_desc: TensorDescriptor,
        scale: driver.CudaSlice(T),
        scale_diff: driver.CudaSlice(T),
        bias_diff: driver.CudaSlice(T),
        epsilon: f64,
        saved_mean: ?driver.CudaSlice(T),
        saved_inv_var: ?driver.CudaSlice(T),
    ) CudnnError!void {
        const one: T = 1;
        const zero: T = 0;
        try result.batchNormalizationBackward(
            self.handle,
            mode.toSys(),
            @ptrCast(&one),
            @ptrCast(&zero),
            @ptrCast(&one),
            @ptrCast(&zero),
            x_desc.desc,
            @ptrFromInt(x.ptr),
            dy_desc.desc,
            @ptrFromInt(dy.ptr),
            dx_desc.desc,
            @ptrFromInt(dx.ptr),
            bn_desc.desc,
            @ptrFromInt(scale.ptr),
            @ptrFromInt(scale_diff.ptr),
            @ptrFromInt(bias_diff.ptr),
            epsilon,
            if (saved_mean) |m| @ptrFromInt(m.ptr) else null,
            if (saved_inv_var) |v| @ptrFromInt(v.ptr) else null,
        );
    }

    // --- Dropout ---

    /// Create a dropout descriptor.
    pub fn createDropout(
        self: Self,
        dropout_ratio: f32,
        states: driver.CudaSlice(u8),
        seed: u64,
    ) CudnnError!DropoutDescriptor {
        const desc = try result.createDropoutDescriptor();
        try result.setDropoutDescriptor(desc, self.handle, dropout_ratio, @ptrFromInt(states.ptr), states.len, seed);
        return .{ .desc = desc };
    }

    /// Get the size of dropout states buffer in bytes.
    pub fn dropoutStatesSize(self: Self) CudnnError!usize {
        return result.dropoutGetStatesSize(self.handle);
    }

    /// Dropout forward: y = dropout(x).
    pub fn dropoutForward(
        self: Self,
        comptime T: type,
        dropout_desc: DropoutDescriptor,
        x_desc: TensorDescriptor,
        x: driver.CudaSlice(T),
        y_desc: TensorDescriptor,
        y: driver.CudaSlice(T),
        reserve: driver.CudaSlice(u8),
    ) CudnnError!void {
        try result.dropoutForward(
            self.handle,
            dropout_desc.desc,
            x_desc.desc,
            @ptrFromInt(x.ptr),
            y_desc.desc,
            @ptrFromInt(y.ptr),
            @ptrFromInt(reserve.ptr),
            reserve.len,
        );
    }

    /// Dropout backward: compute dx from dy.
    pub fn dropoutBackward(
        self: Self,
        comptime T: type,
        dropout_desc: DropoutDescriptor,
        dy_desc: TensorDescriptor,
        dy: driver.CudaSlice(T),
        dx_desc: TensorDescriptor,
        dx: driver.CudaSlice(T),
        reserve: driver.CudaSlice(u8),
    ) CudnnError!void {
        try result.dropoutBackward(
            self.handle,
            dropout_desc.desc,
            dy_desc.desc,
            @ptrFromInt(dy.ptr),
            dx_desc.desc,
            @ptrFromInt(dx.ptr),
            @ptrFromInt(reserve.ptr),
            reserve.len,
        );
    }
};

// ============================================================================
// Tests
// ============================================================================
