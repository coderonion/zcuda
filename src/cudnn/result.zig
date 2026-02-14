/// zCUDA: cuDNN API - Error wrapping layer.
///
/// Layer 2: Converts cuDNN status codes to Zig error unions.
const std = @import("std");
const sys = @import("sys.zig");

// ============================================================================
// Error Type
// ============================================================================

pub const CudnnError = error{
    NotInitialized,
    AllocFailed,
    BadParam,
    InternalError,
    InvalidValue,
    ArchMismatch,
    MappingError,
    ExecutionFailed,
    NotSupported,
    Unknown,
};

pub fn toError(status: sys.cudnnStatus_t) CudnnError!void {
    return switch (status) {
        sys.CUDNN_STATUS_SUCCESS => {},
        sys.CUDNN_STATUS_NOT_INITIALIZED => CudnnError.NotInitialized,
        sys.CUDNN_STATUS_ALLOC_FAILED => CudnnError.AllocFailed,
        sys.CUDNN_STATUS_BAD_PARAM => CudnnError.BadParam,
        sys.CUDNN_STATUS_INTERNAL_ERROR => CudnnError.InternalError,
        sys.CUDNN_STATUS_INVALID_VALUE => CudnnError.InvalidValue,
        sys.CUDNN_STATUS_ARCH_MISMATCH => CudnnError.ArchMismatch,
        sys.CUDNN_STATUS_MAPPING_ERROR => CudnnError.MappingError,
        sys.CUDNN_STATUS_EXECUTION_FAILED => CudnnError.ExecutionFailed,
        sys.CUDNN_STATUS_NOT_SUPPORTED => CudnnError.NotSupported,
        else => CudnnError.Unknown,
    };
}

// ============================================================================
// Handle Management
// ============================================================================

pub fn create() CudnnError!sys.cudnnHandle_t {
    var handle: sys.cudnnHandle_t = undefined;
    try toError(sys.cudnnCreate(&handle));
    return handle;
}

pub fn destroy(handle: sys.cudnnHandle_t) CudnnError!void {
    try toError(sys.cudnnDestroy(handle));
}

pub fn setStream(handle: sys.cudnnHandle_t, stream: ?*anyopaque) CudnnError!void {
    try toError(sys.cudnnSetStream(handle, @ptrCast(stream)));
}

pub fn getVersion() usize {
    return sys.cudnnGetVersion();
}

// ============================================================================
// Tensor Descriptor
// ============================================================================

pub fn createTensorDescriptor() CudnnError!sys.cudnnTensorDescriptor_t {
    var desc: sys.cudnnTensorDescriptor_t = undefined;
    try toError(sys.cudnnCreateTensorDescriptor(&desc));
    return desc;
}

pub fn destroyTensorDescriptor(desc: sys.cudnnTensorDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyTensorDescriptor(desc));
}

pub fn setTensor4dDescriptor(
    desc: sys.cudnnTensorDescriptor_t,
    format: sys.cudnnTensorFormat_t,
    data_type: sys.cudnnDataType_t,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
) CudnnError!void {
    try toError(sys.cudnnSetTensor4dDescriptor(desc, format, data_type, n, c, h, w));
}

/// Set an N-dimensional tensor descriptor (for 3D conv: 5D tensors).
pub fn setTensorNdDescriptor(
    desc: sys.cudnnTensorDescriptor_t,
    data_type: sys.cudnnDataType_t,
    nb_dims: i32,
    dim_a: [*c]const i32,
    stride_a: [*c]const i32,
) CudnnError!void {
    try toError(sys.cudnnSetTensorNdDescriptor(desc, data_type, nb_dims, dim_a, stride_a));
}

// ============================================================================
// Filter Descriptor
// ============================================================================

pub fn createFilterDescriptor() CudnnError!sys.cudnnFilterDescriptor_t {
    var desc: sys.cudnnFilterDescriptor_t = undefined;
    try toError(sys.cudnnCreateFilterDescriptor(&desc));
    return desc;
}

pub fn destroyFilterDescriptor(desc: sys.cudnnFilterDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyFilterDescriptor(desc));
}

pub fn setFilter4dDescriptor(
    desc: sys.cudnnFilterDescriptor_t,
    data_type: sys.cudnnDataType_t,
    format: sys.cudnnTensorFormat_t,
    k: i32,
    c: i32,
    h: i32,
    w: i32,
) CudnnError!void {
    try toError(sys.cudnnSetFilter4dDescriptor(desc, data_type, format, k, c, h, w));
}

/// Set an N-dimensional filter descriptor (for 3D conv: 5D filters).
pub fn setFilterNdDescriptor(
    desc: sys.cudnnFilterDescriptor_t,
    data_type: sys.cudnnDataType_t,
    format: sys.cudnnTensorFormat_t,
    nb_dims: i32,
    filter_dim_a: [*c]const i32,
) CudnnError!void {
    try toError(sys.cudnnSetFilterNdDescriptor(desc, data_type, format, nb_dims, filter_dim_a));
}

// ============================================================================
// Convolution Descriptor
// ============================================================================

pub fn createConvolutionDescriptor() CudnnError!sys.cudnnConvolutionDescriptor_t {
    var desc: sys.cudnnConvolutionDescriptor_t = undefined;
    try toError(sys.cudnnCreateConvolutionDescriptor(&desc));
    return desc;
}

pub fn destroyConvolutionDescriptor(desc: sys.cudnnConvolutionDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyConvolutionDescriptor(desc));
}

pub fn setConvolution2dDescriptor(
    desc: sys.cudnnConvolutionDescriptor_t,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
    mode: sys.cudnnConvolutionMode_t,
    data_type: sys.cudnnDataType_t,
) CudnnError!void {
    try toError(sys.cudnnSetConvolution2dDescriptor(
        desc,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        mode,
        data_type,
    ));
}

pub fn getConvolution2dForwardOutputDim(
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    input_desc: sys.cudnnTensorDescriptor_t,
    filter_desc: sys.cudnnFilterDescriptor_t,
) CudnnError!struct { n: i32, c: i32, h: i32, w: i32 } {
    var n: i32 = undefined;
    var c: i32 = undefined;
    var h: i32 = undefined;
    var w: i32 = undefined;
    try toError(sys.cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w));
    return .{ .n = n, .c = c, .h = h, .w = w };
}

/// Set an N-dimensional convolution descriptor (for 3D conv).
pub fn setConvolutionNdDescriptor(
    desc: sys.cudnnConvolutionDescriptor_t,
    array_length: i32,
    pad_a: [*c]const i32,
    filter_stride_a: [*c]const i32,
    dilation_a: [*c]const i32,
    mode: sys.cudnnConvolutionMode_t,
    data_type: sys.cudnnDataType_t,
) CudnnError!void {
    try toError(sys.cudnnSetConvolutionNdDescriptor(
        desc,
        array_length,
        pad_a,
        filter_stride_a,
        dilation_a,
        mode,
        data_type,
    ));
}

/// Get N-dimensional convolution forward output dimensions.
pub fn getConvolutionNdForwardOutputDim(
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    input_desc: sys.cudnnTensorDescriptor_t,
    filter_desc: sys.cudnnFilterDescriptor_t,
    nb_dims: i32,
    tensor_output_dim_a: [*c]i32,
) CudnnError!void {
    try toError(sys.cudnnGetConvolutionNdForwardOutputDim(
        conv_desc,
        input_desc,
        filter_desc,
        nb_dims,
        tensor_output_dim_a,
    ));
}

pub fn convolutionForward(
    handle: sys.cudnnHandle_t,
    alpha: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    w_desc: sys.cudnnFilterDescriptor_t,
    w: *const anyopaque,
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    algo: sys.cudnnConvolutionFwdAlgo_t,
    workspace: ?*anyopaque,
    workspace_size: usize,
    beta: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnConvolutionForward(
        handle,
        alpha,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        workspace,
        workspace_size,
        beta,
        y_desc,
        y,
    ));
}

pub fn getConvolutionForwardWorkspaceSize(
    handle: sys.cudnnHandle_t,
    x_desc: sys.cudnnTensorDescriptor_t,
    w_desc: sys.cudnnFilterDescriptor_t,
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    y_desc: sys.cudnnTensorDescriptor_t,
    algo: sys.cudnnConvolutionFwdAlgo_t,
) CudnnError!usize {
    var size: usize = undefined;
    try toError(sys.cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc, y_desc, algo, &size));
    return size;
}

// ============================================================================
// Convolution Backward Data
// ============================================================================

pub fn convolutionBackwardData(
    handle: sys.cudnnHandle_t,
    alpha: *const anyopaque,
    w_desc: sys.cudnnFilterDescriptor_t,
    w: *const anyopaque,
    dy_desc: sys.cudnnTensorDescriptor_t,
    dy: *const anyopaque,
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    algo: sys.cudnnConvolutionBwdDataAlgo_t,
    workspace: ?*anyopaque,
    workspace_size: usize,
    beta: *const anyopaque,
    dx_desc: sys.cudnnTensorDescriptor_t,
    dx: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnConvolutionBackwardData(
        handle,
        alpha,
        w_desc,
        w,
        dy_desc,
        dy,
        conv_desc,
        algo,
        workspace,
        workspace_size,
        beta,
        dx_desc,
        dx,
    ));
}

pub fn getConvolutionBackwardDataWorkspaceSize(
    handle: sys.cudnnHandle_t,
    w_desc: sys.cudnnFilterDescriptor_t,
    dy_desc: sys.cudnnTensorDescriptor_t,
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    dx_desc: sys.cudnnTensorDescriptor_t,
    algo: sys.cudnnConvolutionBwdDataAlgo_t,
) CudnnError!usize {
    var size: usize = undefined;
    try toError(sys.cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, dy_desc, conv_desc, dx_desc, algo, &size));
    return size;
}

// ============================================================================
// Convolution Backward Filter
// ============================================================================

pub fn convolutionBackwardFilter(
    handle: sys.cudnnHandle_t,
    alpha: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    dy_desc: sys.cudnnTensorDescriptor_t,
    dy: *const anyopaque,
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    algo: sys.cudnnConvolutionBwdFilterAlgo_t,
    workspace: ?*anyopaque,
    workspace_size: usize,
    beta: *const anyopaque,
    dw_desc: sys.cudnnFilterDescriptor_t,
    dw: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnConvolutionBackwardFilter(
        handle,
        alpha,
        x_desc,
        x,
        dy_desc,
        dy,
        conv_desc,
        algo,
        workspace,
        workspace_size,
        beta,
        dw_desc,
        dw,
    ));
}

pub fn getConvolutionBackwardFilterWorkspaceSize(
    handle: sys.cudnnHandle_t,
    x_desc: sys.cudnnTensorDescriptor_t,
    dy_desc: sys.cudnnTensorDescriptor_t,
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    dw_desc: sys.cudnnFilterDescriptor_t,
    algo: sys.cudnnConvolutionBwdFilterAlgo_t,
) CudnnError!usize {
    var size: usize = undefined;
    try toError(sys.cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, dy_desc, conv_desc, dw_desc, algo, &size));
    return size;
}

// ============================================================================
// Fused Conv + Bias + Activation
// ============================================================================

pub fn convolutionBiasActivationForward(
    handle: sys.cudnnHandle_t,
    alpha1: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    w_desc: sys.cudnnFilterDescriptor_t,
    w: *const anyopaque,
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    algo: sys.cudnnConvolutionFwdAlgo_t,
    workspace: ?*anyopaque,
    workspace_size: usize,
    alpha2: *const anyopaque,
    z_desc: sys.cudnnTensorDescriptor_t,
    z: *const anyopaque,
    bias_desc: sys.cudnnTensorDescriptor_t,
    bias: *const anyopaque,
    act_desc: sys.cudnnActivationDescriptor_t,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnConvolutionBiasActivationForward(
        handle,
        alpha1,
        x_desc,
        x,
        w_desc,
        w,
        conv_desc,
        algo,
        workspace,
        workspace_size,
        alpha2,
        z_desc,
        z,
        bias_desc,
        bias,
        act_desc,
        y_desc,
        y,
    ));
}

// ============================================================================
// Activation Backward
// ============================================================================

pub fn activationBackward(
    handle: sys.cudnnHandle_t,
    activation_desc: sys.cudnnActivationDescriptor_t,
    alpha: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *const anyopaque,
    dy_desc: sys.cudnnTensorDescriptor_t,
    dy: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    beta: *const anyopaque,
    dx_desc: sys.cudnnTensorDescriptor_t,
    dx: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnActivationBackward(
        handle,
        activation_desc,
        alpha,
        y_desc,
        y,
        dy_desc,
        dy,
        x_desc,
        x,
        beta,
        dx_desc,
        dx,
    ));
}

// ============================================================================
// Activation Descriptor
// ============================================================================

pub fn createActivationDescriptor() CudnnError!sys.cudnnActivationDescriptor_t {
    var desc: sys.cudnnActivationDescriptor_t = undefined;
    try toError(sys.cudnnCreateActivationDescriptor(&desc));
    return desc;
}

pub fn destroyActivationDescriptor(desc: sys.cudnnActivationDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyActivationDescriptor(desc));
}

pub fn setActivationDescriptor(
    desc: sys.cudnnActivationDescriptor_t,
    mode: sys.cudnnActivationMode_t,
    nan_prop: sys.cudnnNanPropagation_t,
    coef: f64,
) CudnnError!void {
    try toError(sys.cudnnSetActivationDescriptor(desc, mode, nan_prop, coef));
}

pub fn activationForward(
    handle: sys.cudnnHandle_t,
    activation_desc: sys.cudnnActivationDescriptor_t,
    alpha: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    beta: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnActivationForward(handle, activation_desc, alpha, x_desc, x, beta, y_desc, y));
}

// ============================================================================
// Pooling Descriptor
// ============================================================================

pub fn createPoolingDescriptor() CudnnError!sys.cudnnPoolingDescriptor_t {
    var desc: sys.cudnnPoolingDescriptor_t = undefined;
    try toError(sys.cudnnCreatePoolingDescriptor(&desc));
    return desc;
}

pub fn destroyPoolingDescriptor(desc: sys.cudnnPoolingDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyPoolingDescriptor(desc));
}

pub fn setPooling2dDescriptor(
    desc: sys.cudnnPoolingDescriptor_t,
    mode: sys.cudnnPoolingMode_t,
    nan_prop: sys.cudnnNanPropagation_t,
    window_h: i32,
    window_w: i32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
) CudnnError!void {
    try toError(sys.cudnnSetPooling2dDescriptor(desc, mode, nan_prop, window_h, window_w, pad_h, pad_w, stride_h, stride_w));
}

pub fn poolingForward(
    handle: sys.cudnnHandle_t,
    pooling_desc: sys.cudnnPoolingDescriptor_t,
    alpha: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    beta: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnPoolingForward(handle, pooling_desc, alpha, x_desc, x, beta, y_desc, y));
}

pub fn poolingBackward(
    handle: sys.cudnnHandle_t,
    pooling_desc: sys.cudnnPoolingDescriptor_t,
    alpha: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *const anyopaque,
    dy_desc: sys.cudnnTensorDescriptor_t,
    dy: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    beta: *const anyopaque,
    dx_desc: sys.cudnnTensorDescriptor_t,
    dx: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnPoolingBackward(
        handle,
        pooling_desc,
        alpha,
        y_desc,
        y,
        dy_desc,
        dy,
        x_desc,
        x,
        beta,
        dx_desc,
        dx,
    ));
}

// ============================================================================
// Softmax
// ============================================================================

pub fn softmaxForward(
    handle: sys.cudnnHandle_t,
    algo: sys.cudnnSoftmaxAlgorithm_t,
    mode: sys.cudnnSoftmaxMode_t,
    alpha: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    beta: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnSoftmaxForward(handle, algo, mode, alpha, x_desc, x, beta, y_desc, y));
}

pub fn softmaxBackward(
    handle: sys.cudnnHandle_t,
    algo: sys.cudnnSoftmaxAlgorithm_t,
    mode: sys.cudnnSoftmaxMode_t,
    alpha: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *const anyopaque,
    dy_desc: sys.cudnnTensorDescriptor_t,
    dy: *const anyopaque,
    beta: *const anyopaque,
    dx_desc: sys.cudnnTensorDescriptor_t,
    dx: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnSoftmaxBackward(handle, algo, mode, alpha, y_desc, y, dy_desc, dy, beta, dx_desc, dx));
}

// ============================================================================
// Reduction
// ============================================================================

pub fn createReduceTensorDescriptor() CudnnError!sys.cudnnReduceTensorDescriptor_t {
    var desc: sys.cudnnReduceTensorDescriptor_t = undefined;
    try toError(sys.cudnnCreateReduceTensorDescriptor(&desc));
    return desc;
}

pub fn destroyReduceTensorDescriptor(desc: sys.cudnnReduceTensorDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyReduceTensorDescriptor(desc));
}

pub fn setReduceTensorDescriptor(
    desc: sys.cudnnReduceTensorDescriptor_t,
    op: sys.cudnnReduceTensorOp_t,
    comp_type: sys.cudnnDataType_t,
    nan_opt: sys.cudnnNanPropagation_t,
    indices: sys.cudnnReduceTensorIndices_t,
    indices_type: sys.cudnnIndicesType_t,
) CudnnError!void {
    try toError(sys.cudnnSetReduceTensorDescriptor(desc, op, comp_type, nan_opt, indices, indices_type));
}

pub fn reduceTensor(
    handle: sys.cudnnHandle_t,
    reduce_desc: sys.cudnnReduceTensorDescriptor_t,
    indices: ?*anyopaque,
    indices_size: usize,
    workspace: ?*anyopaque,
    workspace_size: usize,
    alpha: *const anyopaque,
    a_desc: sys.cudnnTensorDescriptor_t,
    a: *const anyopaque,
    beta: *const anyopaque,
    c_desc: sys.cudnnTensorDescriptor_t,
    c: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnReduceTensor(
        handle,
        reduce_desc,
        indices,
        indices_size,
        workspace,
        workspace_size,
        alpha,
        a_desc,
        a,
        beta,
        c_desc,
        c,
    ));
}

pub fn getReductionWorkspaceSize(
    handle: sys.cudnnHandle_t,
    reduce_desc: sys.cudnnReduceTensorDescriptor_t,
    a_desc: sys.cudnnTensorDescriptor_t,
    c_desc: sys.cudnnTensorDescriptor_t,
) CudnnError!usize {
    var size: usize = undefined;
    try toError(sys.cudnnGetReductionWorkspaceSize(handle, reduce_desc, a_desc, c_desc, &size));
    return size;
}

// ============================================================================
// Batch Normalization
// ============================================================================

/// Derive the tensor descriptor for BN scale/bias/mean/variance from the input.
pub fn deriveBNTensorDescriptor(
    derived_desc: sys.cudnnTensorDescriptor_t,
    x_desc: sys.cudnnTensorDescriptor_t,
    mode: sys.cudnnBatchNormMode_t,
) CudnnError!void {
    try toError(sys.cudnnDeriveBNTensorDescriptor(derived_desc, x_desc, mode));
}

/// Batch normalization forward (training mode).
pub fn batchNormalizationForwardTraining(
    handle: sys.cudnnHandle_t,
    mode: sys.cudnnBatchNormMode_t,
    alpha: *const anyopaque,
    beta: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
    bn_scale_bias_mean_var_desc: sys.cudnnTensorDescriptor_t,
    bn_scale: *const anyopaque,
    bn_bias: *const anyopaque,
    exp_avg_factor: f64,
    result_running_mean: ?*anyopaque,
    result_running_var: ?*anyopaque,
    epsilon: f64,
    result_save_mean: ?*anyopaque,
    result_save_inv_variance: ?*anyopaque,
) CudnnError!void {
    try toError(sys.cudnnBatchNormalizationForwardTraining(
        handle,
        mode,
        alpha,
        beta,
        x_desc,
        x,
        y_desc,
        y,
        bn_scale_bias_mean_var_desc,
        bn_scale,
        bn_bias,
        exp_avg_factor,
        result_running_mean,
        result_running_var,
        epsilon,
        result_save_mean,
        result_save_inv_variance,
    ));
}

/// Batch normalization forward (inference mode).
pub fn batchNormalizationForwardInference(
    handle: sys.cudnnHandle_t,
    mode: sys.cudnnBatchNormMode_t,
    alpha: *const anyopaque,
    beta: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
    bn_scale_bias_mean_var_desc: sys.cudnnTensorDescriptor_t,
    bn_scale: *const anyopaque,
    bn_bias: *const anyopaque,
    estimated_mean: *const anyopaque,
    estimated_variance: *const anyopaque,
    epsilon: f64,
) CudnnError!void {
    try toError(sys.cudnnBatchNormalizationForwardInference(
        handle,
        mode,
        alpha,
        beta,
        x_desc,
        x,
        y_desc,
        y,
        bn_scale_bias_mean_var_desc,
        bn_scale,
        bn_bias,
        estimated_mean,
        estimated_variance,
        epsilon,
    ));
}

/// Batch normalization backward.
pub fn batchNormalizationBackward(
    handle: sys.cudnnHandle_t,
    mode: sys.cudnnBatchNormMode_t,
    alpha_data_diff: *const anyopaque,
    beta_data_diff: *const anyopaque,
    alpha_param_diff: *const anyopaque,
    beta_param_diff: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    dy_desc: sys.cudnnTensorDescriptor_t,
    dy: *const anyopaque,
    dx_desc: sys.cudnnTensorDescriptor_t,
    dx: *anyopaque,
    bn_scale_bias_desc: sys.cudnnTensorDescriptor_t,
    bn_scale: *const anyopaque,
    result_bn_scale_diff: *anyopaque,
    result_bn_bias_diff: *anyopaque,
    epsilon: f64,
    saved_mean: ?*const anyopaque,
    saved_inv_variance: ?*const anyopaque,
) CudnnError!void {
    try toError(sys.cudnnBatchNormalizationBackward(
        handle,
        mode,
        alpha_data_diff,
        beta_data_diff,
        alpha_param_diff,
        beta_param_diff,
        x_desc,
        x,
        dy_desc,
        dy,
        dx_desc,
        dx,
        bn_scale_bias_desc,
        bn_scale,
        result_bn_scale_diff,
        result_bn_bias_diff,
        epsilon,
        saved_mean,
        saved_inv_variance,
    ));
}

// ============================================================================
// Dropout
// ============================================================================

pub fn createDropoutDescriptor() CudnnError!sys.cudnnDropoutDescriptor_t {
    var desc: sys.cudnnDropoutDescriptor_t = undefined;
    try toError(sys.cudnnCreateDropoutDescriptor(&desc));
    return desc;
}

pub fn destroyDropoutDescriptor(desc: sys.cudnnDropoutDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyDropoutDescriptor(desc));
}

pub fn dropoutGetStatesSize(handle: sys.cudnnHandle_t) CudnnError!usize {
    var size: usize = undefined;
    try toError(sys.cudnnDropoutGetStatesSize(handle, &size));
    return size;
}

pub fn dropoutGetReserveSpaceSize(x_desc: sys.cudnnTensorDescriptor_t) CudnnError!usize {
    var size: usize = undefined;
    try toError(sys.cudnnDropoutGetReserveSpaceSize(x_desc, &size));
    return size;
}

pub fn setDropoutDescriptor(
    desc: sys.cudnnDropoutDescriptor_t,
    handle: sys.cudnnHandle_t,
    dropout_ratio: f32,
    states: *anyopaque,
    states_size: usize,
    seed: u64,
) CudnnError!void {
    try toError(sys.cudnnSetDropoutDescriptor(desc, handle, dropout_ratio, states, states_size, seed));
}

pub fn dropoutForward(
    handle: sys.cudnnHandle_t,
    dropout_desc: sys.cudnnDropoutDescriptor_t,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
    reserve_space: *anyopaque,
    reserve_space_size: usize,
) CudnnError!void {
    try toError(sys.cudnnDropoutForward(
        handle,
        dropout_desc,
        x_desc,
        x,
        y_desc,
        y,
        reserve_space,
        reserve_space_size,
    ));
}

pub fn dropoutBackward(
    handle: sys.cudnnHandle_t,
    dropout_desc: sys.cudnnDropoutDescriptor_t,
    dy_desc: sys.cudnnTensorDescriptor_t,
    dy: *const anyopaque,
    dx_desc: sys.cudnnTensorDescriptor_t,
    dx: *anyopaque,
    reserve_space: *anyopaque,
    reserve_space_size: usize,
) CudnnError!void {
    try toError(sys.cudnnDropoutBackward(
        handle,
        dropout_desc,
        dy_desc,
        dy,
        dx_desc,
        dx,
        reserve_space,
        reserve_space_size,
    ));
}

// ============================================================================
// Tensor Operations (P0 Critical)
// ============================================================================

/// Add a tensor to another: C = alpha*A + beta*C (bias add).
pub fn addTensor(
    handle: sys.cudnnHandle_t,
    alpha: *const anyopaque,
    a_desc: sys.cudnnTensorDescriptor_t,
    a: *const anyopaque,
    beta: *const anyopaque,
    c_desc: sys.cudnnTensorDescriptor_t,
    c_out: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnAddTensor(handle, alpha, a_desc, a, beta, c_desc, c_out));
}

/// Scale a tensor in-place: Y = alpha * Y.
pub fn scaleTensor(
    handle: sys.cudnnHandle_t,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
    alpha: *const anyopaque,
) CudnnError!void {
    try toError(sys.cudnnScaleTensor(handle, y_desc, y, alpha));
}

/// Transform a tensor (layout conversion, e.g. NCHW ↔ NHWC).
pub fn transformTensor(
    handle: sys.cudnnHandle_t,
    alpha: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    beta: *const anyopaque,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnTransformTensor(handle, alpha, x_desc, x, beta, y_desc, y));
}

/// Set group count for grouped/depthwise convolution.
pub fn setConvolutionGroupCount(
    conv_desc: sys.cudnnConvolutionDescriptor_t,
    group_count: i32,
) CudnnError!void {
    try toError(sys.cudnnSetConvolutionGroupCount(conv_desc, group_count));
}

// ============================================================================
// Element-wise Operations (P1)
// ============================================================================

/// Element-wise tensor operation: C = op(alpha1*A, alpha2*B) + beta*C.
pub fn opTensor(
    handle: sys.cudnnHandle_t,
    op_desc: sys.cudnnOpTensorDescriptor_t,
    alpha1: *const anyopaque,
    a_desc: sys.cudnnTensorDescriptor_t,
    a: *const anyopaque,
    alpha2: *const anyopaque,
    b_desc: sys.cudnnTensorDescriptor_t,
    b: *const anyopaque,
    beta: *const anyopaque,
    c_desc: sys.cudnnTensorDescriptor_t,
    c_out: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnOpTensor(handle, op_desc, alpha1, a_desc, a, alpha2, b_desc, b, beta, c_desc, c_out));
}

/// Create an op tensor descriptor.
pub fn createOpTensorDescriptor() CudnnError!sys.cudnnOpTensorDescriptor_t {
    var desc: sys.cudnnOpTensorDescriptor_t = undefined;
    try toError(sys.cudnnCreateOpTensorDescriptor(&desc));
    return desc;
}

/// Destroy an op tensor descriptor.
pub fn destroyOpTensorDescriptor(desc: sys.cudnnOpTensorDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyOpTensorDescriptor(desc));
}

/// Set an op tensor descriptor.
pub fn setOpTensorDescriptor(
    desc: sys.cudnnOpTensorDescriptor_t,
    op: sys.cudnnOpTensorOp_t,
    comp_type: sys.cudnnDataType_t,
    nan_prop: sys.cudnnNanPropagation_t,
) CudnnError!void {
    try toError(sys.cudnnSetOpTensorDescriptor(desc, op, comp_type, nan_prop));
}

// ============================================================================
// Convolution Backward Bias
// ============================================================================

/// Compute the gradient of the bias during backpropagation.
/// db = alpha * sum(dy) + beta * db
pub fn convolutionBackwardBias(
    handle: sys.cudnnHandle_t,
    alpha: *const anyopaque,
    dy_desc: sys.cudnnTensorDescriptor_t,
    dy: *const anyopaque,
    beta: *const anyopaque,
    db_desc: sys.cudnnTensorDescriptor_t,
    db: *anyopaque,
) CudnnError!void {
    try toError(sys.cudnnConvolutionBackwardBias(handle, alpha, dy_desc, dy, beta, db_desc, db));
}

// ============================================================================
// Normalization (LayerNorm/GroupNorm)
// ============================================================================

/// Normalization forward pass for inference.
pub fn normalizationForwardInference(
    handle: sys.cudnnHandle_t,
    mode: sys.cudnnNormMode_t,
    norm_ops: sys.cudnnNormOps_t,
    algo: sys.cudnnNormAlgo_t,
    alpha: *const anyopaque,
    beta: *const anyopaque,
    x_desc: sys.cudnnTensorDescriptor_t,
    x: *const anyopaque,
    norm_scale_bias_desc: sys.cudnnTensorDescriptor_t,
    norm_scale: *const anyopaque,
    norm_bias: *const anyopaque,
    norm_mean_var_desc: sys.cudnnTensorDescriptor_t,
    estimated_mean: *const anyopaque,
    estimated_variance: *const anyopaque,
    z_desc: sys.cudnnTensorDescriptor_t,
    z: ?*const anyopaque,
    activation_desc: sys.cudnnActivationDescriptor_t,
    y_desc: sys.cudnnTensorDescriptor_t,
    y: *anyopaque,
    epsilon: f64,
    group_cnt: c_int,
) CudnnError!void {
    try toError(sys.cudnnNormalizationForwardInference(
        handle,
        mode,
        norm_ops,
        algo,
        alpha,
        beta,
        x_desc,
        x,
        norm_scale_bias_desc,
        norm_scale,
        norm_bias,
        norm_mean_var_desc,
        estimated_mean,
        estimated_variance,
        z_desc,
        z,
        activation_desc,
        y_desc,
        y,
        epsilon,
        group_cnt,
    ));
}

// ============================================================================
// RNN — Descriptor Management
// ============================================================================

pub fn createRNNDescriptor() CudnnError!sys.cudnnRNNDescriptor_t {
    var desc: sys.cudnnRNNDescriptor_t = undefined;
    try toError(sys.cudnnCreateRNNDescriptor(&desc));
    return desc;
}

pub fn destroyRNNDescriptor(desc: sys.cudnnRNNDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyRNNDescriptor(desc));
}

pub fn setRNNDescriptor_v8(
    desc: sys.cudnnRNNDescriptor_t,
    algo: sys.cudnnRNNAlgo_t,
    cell_mode: sys.cudnnRNNMode_t,
    bias_mode: sys.cudnnRNNBiasMode_t,
    dir_mode: sys.cudnnDirectionMode_t,
    input_mode: sys.cudnnInputMode_t,
    data_type: sys.cudnnDataType_t,
    math_prec: sys.cudnnDataType_t,
    math_type: sys.cudnnMathType_t,
    input_size: i32,
    hidden_size: i32,
    proj_size: i32,
    num_layers: i32,
    dropout_desc: sys.cudnnDropoutDescriptor_t,
    aux_flags: u32,
) CudnnError!void {
    try toError(sys.cudnnSetRNNDescriptor_v8(
        desc,
        algo,
        cell_mode,
        bias_mode,
        dir_mode,
        input_mode,
        data_type,
        math_prec,
        math_type,
        input_size,
        hidden_size,
        proj_size,
        num_layers,
        dropout_desc,
        aux_flags,
    ));
}

// RNN Data Descriptor
pub fn createRNNDataDescriptor() CudnnError!sys.cudnnRNNDataDescriptor_t {
    var desc: sys.cudnnRNNDataDescriptor_t = undefined;
    try toError(sys.cudnnCreateRNNDataDescriptor(&desc));
    return desc;
}

pub fn destroyRNNDataDescriptor(desc: sys.cudnnRNNDataDescriptor_t) CudnnError!void {
    try toError(sys.cudnnDestroyRNNDataDescriptor(desc));
}

// RNN Workspace
pub fn getRNNWeightSpaceSize(handle: sys.cudnnHandle_t, rnn_desc: sys.cudnnRNNDescriptor_t) CudnnError!usize {
    var size: usize = undefined;
    try toError(sys.cudnnGetRNNWeightSpaceSize(handle, rnn_desc, &size));
    return size;
}

// RNN Forward
pub fn rnnForward(
    handle: sys.cudnnHandle_t,
    rnn_desc: sys.cudnnRNNDescriptor_t,
    fwd_mode: sys.cudnnForwardMode_t,
    dev_seq_lengths: [*c]const i32,
    x_desc: sys.cudnnRNNDataDescriptor_t,
    x: *const anyopaque,
    y_desc: sys.cudnnRNNDataDescriptor_t,
    y: *anyopaque,
    h_desc: sys.cudnnTensorDescriptor_t,
    hx: ?*const anyopaque,
    hy: ?*anyopaque,
    c_desc: sys.cudnnTensorDescriptor_t,
    cx: ?*const anyopaque,
    cy: ?*anyopaque,
    weight_space_size: usize,
    weight_space: *const anyopaque,
    work_space_size: usize,
    work_space: *anyopaque,
    reserve_space_size: usize,
    reserve_space: ?*anyopaque,
) CudnnError!void {
    try toError(sys.cudnnRNNForward(
        handle,
        rnn_desc,
        fwd_mode,
        dev_seq_lengths,
        x_desc,
        x,
        y_desc,
        y,
        h_desc,
        hx,
        hy,
        c_desc,
        cx,
        cy,
        weight_space_size,
        weight_space,
        work_space_size,
        work_space,
        reserve_space_size,
        reserve_space,
    ));
}
