/// zCUDA: cuDNN API - Raw FFI bindings.
///
/// Layer 1: Direct @cImport of cudnn.h for deep learning primitives.
const std = @import("std");

pub const c = @cImport({
    @cInclude("cudnn.h");
});

// Core types
pub const cudnnStatus_t = c.cudnnStatus_t;
pub const cudnnHandle_t = c.cudnnHandle_t;
pub const cudnnTensorDescriptor_t = c.cudnnTensorDescriptor_t;
pub const cudnnFilterDescriptor_t = c.cudnnFilterDescriptor_t;
pub const cudnnConvolutionDescriptor_t = c.cudnnConvolutionDescriptor_t;
pub const cudnnActivationDescriptor_t = c.cudnnActivationDescriptor_t;
pub const cudnnPoolingDescriptor_t = c.cudnnPoolingDescriptor_t;
pub const cudnnReduceTensorDescriptor_t = c.cudnnReduceTensorDescriptor_t;
pub const cudnnDataType_t = c.cudnnDataType_t;
pub const cudnnTensorFormat_t = c.cudnnTensorFormat_t;
pub const cudnnActivationMode_t = c.cudnnActivationMode_t;
pub const cudnnConvolutionMode_t = c.cudnnConvolutionMode_t;
pub const cudnnConvolutionFwdAlgo_t = c.cudnnConvolutionFwdAlgo_t;
pub const cudnnPoolingMode_t = c.cudnnPoolingMode_t;
pub const cudnnSoftmaxAlgorithm_t = c.cudnnSoftmaxAlgorithm_t;
pub const cudnnSoftmaxMode_t = c.cudnnSoftmaxMode_t;
pub const cudnnNanPropagation_t = c.cudnnNanPropagation_t;
pub const cudnnReduceTensorOp_t = c.cudnnReduceTensorOp_t;
pub const cudnnReduceTensorIndices_t = c.cudnnReduceTensorIndices_t;
pub const cudnnIndicesType_t = c.cudnnIndicesType_t;

// Status codes
pub const CUDNN_STATUS_SUCCESS = c.CUDNN_STATUS_SUCCESS;
pub const CUDNN_STATUS_NOT_INITIALIZED = c.CUDNN_STATUS_NOT_INITIALIZED;
pub const CUDNN_STATUS_ALLOC_FAILED = c.CUDNN_STATUS_ALLOC_FAILED;
pub const CUDNN_STATUS_BAD_PARAM = c.CUDNN_STATUS_BAD_PARAM;
pub const CUDNN_STATUS_INTERNAL_ERROR = c.CUDNN_STATUS_INTERNAL_ERROR;
pub const CUDNN_STATUS_INVALID_VALUE = c.CUDNN_STATUS_INVALID_VALUE;
pub const CUDNN_STATUS_ARCH_MISMATCH = c.CUDNN_STATUS_ARCH_MISMATCH;
pub const CUDNN_STATUS_MAPPING_ERROR = c.CUDNN_STATUS_MAPPING_ERROR;
pub const CUDNN_STATUS_EXECUTION_FAILED = c.CUDNN_STATUS_EXECUTION_FAILED;
pub const CUDNN_STATUS_NOT_SUPPORTED = c.CUDNN_STATUS_NOT_SUPPORTED;

// Data types
pub const CUDNN_DATA_FLOAT = c.CUDNN_DATA_FLOAT;
pub const CUDNN_DATA_DOUBLE = c.CUDNN_DATA_DOUBLE;
pub const CUDNN_DATA_HALF = c.CUDNN_DATA_HALF;
pub const CUDNN_DATA_BFLOAT16 = c.CUDNN_DATA_BFLOAT16;

// Tensor format
pub const CUDNN_TENSOR_NCHW = c.CUDNN_TENSOR_NCHW;
pub const CUDNN_TENSOR_NHWC = c.CUDNN_TENSOR_NHWC;

// Activation modes
pub const CUDNN_ACTIVATION_SIGMOID = c.CUDNN_ACTIVATION_SIGMOID;
pub const CUDNN_ACTIVATION_RELU = c.CUDNN_ACTIVATION_RELU;
pub const CUDNN_ACTIVATION_TANH = c.CUDNN_ACTIVATION_TANH;
pub const CUDNN_ACTIVATION_CLIPPED_RELU = c.CUDNN_ACTIVATION_CLIPPED_RELU;
pub const CUDNN_ACTIVATION_ELU = c.CUDNN_ACTIVATION_ELU;
pub const CUDNN_ACTIVATION_IDENTITY = c.CUDNN_ACTIVATION_IDENTITY;

// NaN propagation
pub const CUDNN_NOT_PROPAGATE_NAN = c.CUDNN_NOT_PROPAGATE_NAN;
pub const CUDNN_PROPAGATE_NAN = c.CUDNN_PROPAGATE_NAN;

// Convolution modes
pub const CUDNN_CONVOLUTION = c.CUDNN_CONVOLUTION;
pub const CUDNN_CROSS_CORRELATION = c.CUDNN_CROSS_CORRELATION;

// Convolution forward algorithms
pub const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = c.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
pub const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = c.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
pub const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = c.CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
pub const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = c.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
pub const CUDNN_CONVOLUTION_FWD_ALGO_FFT = c.CUDNN_CONVOLUTION_FWD_ALGO_FFT;
pub const CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = c.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
pub const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = c.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
pub const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = c.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;

// Pooling modes
pub const CUDNN_POOLING_MAX = c.CUDNN_POOLING_MAX;
pub const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = c.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
pub const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = c.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
pub const CUDNN_POOLING_MAX_DETERMINISTIC = c.CUDNN_POOLING_MAX_DETERMINISTIC;

// Softmax
pub const CUDNN_SOFTMAX_FAST = c.CUDNN_SOFTMAX_FAST;
pub const CUDNN_SOFTMAX_ACCURATE = c.CUDNN_SOFTMAX_ACCURATE;
pub const CUDNN_SOFTMAX_LOG = c.CUDNN_SOFTMAX_LOG;
pub const CUDNN_SOFTMAX_MODE_INSTANCE = c.CUDNN_SOFTMAX_MODE_INSTANCE;
pub const CUDNN_SOFTMAX_MODE_CHANNEL = c.CUDNN_SOFTMAX_MODE_CHANNEL;

// Reduce tensor
pub const CUDNN_REDUCE_TENSOR_ADD = c.CUDNN_REDUCE_TENSOR_ADD;
pub const CUDNN_REDUCE_TENSOR_MUL = c.CUDNN_REDUCE_TENSOR_MUL;
pub const CUDNN_REDUCE_TENSOR_MIN = c.CUDNN_REDUCE_TENSOR_MIN;
pub const CUDNN_REDUCE_TENSOR_MAX = c.CUDNN_REDUCE_TENSOR_MAX;
pub const CUDNN_REDUCE_TENSOR_AMAX = c.CUDNN_REDUCE_TENSOR_AMAX;
pub const CUDNN_REDUCE_TENSOR_AVG = c.CUDNN_REDUCE_TENSOR_AVG;
pub const CUDNN_REDUCE_TENSOR_NORM1 = c.CUDNN_REDUCE_TENSOR_NORM1;
pub const CUDNN_REDUCE_TENSOR_NORM2 = c.CUDNN_REDUCE_TENSOR_NORM2;
pub const CUDNN_REDUCE_TENSOR_NO_INDICES = c.CUDNN_REDUCE_TENSOR_NO_INDICES;
pub const CUDNN_32BIT_INDICES = c.CUDNN_32BIT_INDICES;

// Core functions
pub const cudnnCreate = c.cudnnCreate;
pub const cudnnDestroy = c.cudnnDestroy;
pub const cudnnSetStream = c.cudnnSetStream;
pub const cudnnGetVersion = c.cudnnGetVersion;

// Tensor descriptor functions
pub const cudnnCreateTensorDescriptor = c.cudnnCreateTensorDescriptor;
pub const cudnnDestroyTensorDescriptor = c.cudnnDestroyTensorDescriptor;
pub const cudnnSetTensor4dDescriptor = c.cudnnSetTensor4dDescriptor;
pub const cudnnSetTensorNdDescriptor = c.cudnnSetTensorNdDescriptor;

// Filter descriptor functions
pub const cudnnCreateFilterDescriptor = c.cudnnCreateFilterDescriptor;
pub const cudnnDestroyFilterDescriptor = c.cudnnDestroyFilterDescriptor;
pub const cudnnSetFilter4dDescriptor = c.cudnnSetFilter4dDescriptor;
pub const cudnnSetFilterNdDescriptor = c.cudnnSetFilterNdDescriptor;

// Convolution descriptor functions
pub const cudnnCreateConvolutionDescriptor = c.cudnnCreateConvolutionDescriptor;
pub const cudnnDestroyConvolutionDescriptor = c.cudnnDestroyConvolutionDescriptor;
pub const cudnnSetConvolution2dDescriptor = c.cudnnSetConvolution2dDescriptor;
pub const cudnnSetConvolutionNdDescriptor = c.cudnnSetConvolutionNdDescriptor;
pub const cudnnGetConvolution2dForwardOutputDim = c.cudnnGetConvolution2dForwardOutputDim;
pub const cudnnGetConvolutionNdForwardOutputDim = c.cudnnGetConvolutionNdForwardOutputDim;
pub const cudnnConvolutionForward = c.cudnnConvolutionForward;
pub const cudnnGetConvolutionForwardWorkspaceSize = c.cudnnGetConvolutionForwardWorkspaceSize;

// Activation functions
pub const cudnnCreateActivationDescriptor = c.cudnnCreateActivationDescriptor;
pub const cudnnDestroyActivationDescriptor = c.cudnnDestroyActivationDescriptor;
pub const cudnnSetActivationDescriptor = c.cudnnSetActivationDescriptor;
pub const cudnnActivationForward = c.cudnnActivationForward;

// Pooling functions
pub const cudnnCreatePoolingDescriptor = c.cudnnCreatePoolingDescriptor;
pub const cudnnDestroyPoolingDescriptor = c.cudnnDestroyPoolingDescriptor;
pub const cudnnSetPooling2dDescriptor = c.cudnnSetPooling2dDescriptor;
pub const cudnnPoolingForward = c.cudnnPoolingForward;

// Softmax functions
pub const cudnnSoftmaxForward = c.cudnnSoftmaxForward;

// Reduce tensor functions
pub const cudnnCreateReduceTensorDescriptor = c.cudnnCreateReduceTensorDescriptor;
pub const cudnnDestroyReduceTensorDescriptor = c.cudnnDestroyReduceTensorDescriptor;
pub const cudnnSetReduceTensorDescriptor = c.cudnnSetReduceTensorDescriptor;
pub const cudnnReduceTensor = c.cudnnReduceTensor;
pub const cudnnGetReductionWorkspaceSize = c.cudnnGetReductionWorkspaceSize;

// Convolution backward types
pub const cudnnConvolutionBwdDataAlgo_t = c.cudnnConvolutionBwdDataAlgo_t;
pub const cudnnConvolutionBwdFilterAlgo_t = c.cudnnConvolutionBwdFilterAlgo_t;
pub const CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = c.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
pub const CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = c.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = c.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
pub const CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = c.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

// Convolution backward functions
pub const cudnnConvolutionBackwardData = c.cudnnConvolutionBackwardData;
pub const cudnnGetConvolutionBackwardDataWorkspaceSize = c.cudnnGetConvolutionBackwardDataWorkspaceSize;
pub const cudnnConvolutionBackwardFilter = c.cudnnConvolutionBackwardFilter;
pub const cudnnGetConvolutionBackwardFilterWorkspaceSize = c.cudnnGetConvolutionBackwardFilterWorkspaceSize;
pub const cudnnConvolutionBackwardBias = c.cudnnConvolutionBackwardBias;

// Fused conv + bias + activation
pub const cudnnConvolutionBiasActivationForward = c.cudnnConvolutionBiasActivationForward;

// Activation backward
pub const cudnnActivationBackward = c.cudnnActivationBackward;

// Pooling backward
pub const cudnnPoolingBackward = c.cudnnPoolingBackward;

// Softmax backward
pub const cudnnSoftmaxBackward = c.cudnnSoftmaxBackward;

// Batch Normalization types
pub const cudnnBatchNormMode_t = c.cudnnBatchNormMode_t;
pub const CUDNN_BATCHNORM_PER_ACTIVATION = c.CUDNN_BATCHNORM_PER_ACTIVATION;
pub const CUDNN_BATCHNORM_SPATIAL = c.CUDNN_BATCHNORM_SPATIAL;
pub const CUDNN_BATCHNORM_SPATIAL_PERSISTENT = c.CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

// Batch Normalization functions
pub const cudnnDeriveBNTensorDescriptor = c.cudnnDeriveBNTensorDescriptor;
pub const cudnnBatchNormalizationForwardTraining = c.cudnnBatchNormalizationForwardTraining;
pub const cudnnBatchNormalizationForwardInference = c.cudnnBatchNormalizationForwardInference;
pub const cudnnBatchNormalizationBackward = c.cudnnBatchNormalizationBackward;

// Dropout types and functions
pub const cudnnDropoutDescriptor_t = c.cudnnDropoutDescriptor_t;
pub const cudnnCreateDropoutDescriptor = c.cudnnCreateDropoutDescriptor;
pub const cudnnDestroyDropoutDescriptor = c.cudnnDestroyDropoutDescriptor;
pub const cudnnSetDropoutDescriptor = c.cudnnSetDropoutDescriptor;
pub const cudnnDropoutGetStatesSize = c.cudnnDropoutGetStatesSize;
pub const cudnnDropoutGetReserveSpaceSize = c.cudnnDropoutGetReserveSpaceSize;
pub const cudnnDropoutForward = c.cudnnDropoutForward;
pub const cudnnDropoutBackward = c.cudnnDropoutBackward;

// Tensor operations
pub const cudnnAddTensor = c.cudnnAddTensor;
pub const cudnnScaleTensor = c.cudnnScaleTensor;
pub const cudnnTransformTensor = c.cudnnTransformTensor;
pub const cudnnSetConvolutionGroupCount = c.cudnnSetConvolutionGroupCount;
pub const cudnnOpTensor = c.cudnnOpTensor;
pub const cudnnOpTensorDescriptor_t = c.cudnnOpTensorDescriptor_t;
pub const cudnnCreateOpTensorDescriptor = c.cudnnCreateOpTensorDescriptor;
pub const cudnnDestroyOpTensorDescriptor = c.cudnnDestroyOpTensorDescriptor;
pub const cudnnSetOpTensorDescriptor = c.cudnnSetOpTensorDescriptor;
pub const cudnnOpTensorOp_t = c.cudnnOpTensorOp_t;

// Normalization
pub const cudnnNormMode_t = c.cudnnNormMode_t;
pub const cudnnNormAlgo_t = c.cudnnNormAlgo_t;
pub const cudnnNormOps_t = c.cudnnNormOps_t;
pub const CUDNN_NORM_PER_CHANNEL = c.CUDNN_NORM_PER_CHANNEL;
pub const CUDNN_NORM_PER_ACTIVATION = c.CUDNN_NORM_PER_ACTIVATION;
pub const cudnnNormalizationForwardInference = c.cudnnNormalizationForwardInference;
pub const cudnnNormalizationForwardTraining = c.cudnnNormalizationForwardTraining;
pub const cudnnNormalizationBackward = c.cudnnNormalizationBackward;

// RNN types
pub const cudnnRNNDescriptor_t = c.cudnnRNNDescriptor_t;
pub const cudnnRNNDataDescriptor_t = c.cudnnRNNDataDescriptor_t;
pub const cudnnRNNMode_t = c.cudnnRNNMode_t;
pub const cudnnRNNBiasMode_t = c.cudnnRNNBiasMode_t;
pub const cudnnRNNAlgo_t = c.cudnnRNNAlgo_t;
pub const cudnnDirectionMode_t = c.cudnnDirectionMode_t;
pub const cudnnInputMode_t = c.cudnnInputMode_t;
pub const cudnnMathType_t = c.cudnnMathType_t;
pub const cudnnForwardMode_t = c.cudnnForwardMode_t;
pub const cudnnRNNDataLayout_t = c.cudnnRNNDataLayout_t;

// RNN descriptor management
pub const cudnnCreateRNNDescriptor = c.cudnnCreateRNNDescriptor;
pub const cudnnDestroyRNNDescriptor = c.cudnnDestroyRNNDescriptor;
pub const cudnnSetRNNDescriptor_v8 = c.cudnnSetRNNDescriptor_v8;

// RNN data descriptor management
pub const cudnnCreateRNNDataDescriptor = c.cudnnCreateRNNDataDescriptor;
pub const cudnnDestroyRNNDataDescriptor = c.cudnnDestroyRNNDataDescriptor;
pub const cudnnSetRNNDataDescriptor = c.cudnnSetRNNDataDescriptor;

// RNN workspace/weights
pub const cudnnGetRNNTempSpaceSizes = c.cudnnGetRNNTempSpaceSizes;
pub const cudnnGetRNNWeightSpaceSize = c.cudnnGetRNNWeightSpaceSize;

// RNN forward/backward
pub const cudnnRNNForward = c.cudnnRNNForward;
pub const cudnnRNNBackwardData_v8 = c.cudnnRNNBackwardData_v8;
pub const cudnnRNNBackwardWeights_v8 = c.cudnnRNNBackwardWeights_v8;

// RNN modes/constants
pub const CUDNN_RNN_RELU = c.CUDNN_RNN_RELU;
pub const CUDNN_RNN_TANH = c.CUDNN_RNN_TANH;
pub const CUDNN_LSTM = c.CUDNN_LSTM;
pub const CUDNN_GRU = c.CUDNN_GRU;
pub const CUDNN_UNIDIRECTIONAL = c.CUDNN_UNIDIRECTIONAL;
pub const CUDNN_BIDIRECTIONAL = c.CUDNN_BIDIRECTIONAL;
pub const CUDNN_LINEAR_INPUT = c.CUDNN_LINEAR_INPUT;
pub const CUDNN_SKIP_INPUT = c.CUDNN_SKIP_INPUT;
pub const CUDNN_RNN_ALGO_STANDARD = c.CUDNN_RNN_ALGO_STANDARD;
pub const CUDNN_RNN_ALGO_PERSIST_STATIC = c.CUDNN_RNN_ALGO_PERSIST_STATIC;
pub const CUDNN_RNN_ALGO_PERSIST_DYNAMIC = c.CUDNN_RNN_ALGO_PERSIST_DYNAMIC;
pub const CUDNN_RNN_DOUBLE_BIAS = c.CUDNN_RNN_DOUBLE_BIAS;
pub const CUDNN_RNN_SINGLE_INP_BIAS = c.CUDNN_RNN_SINGLE_INP_BIAS;
pub const CUDNN_RNN_SINGLE_REC_BIAS = c.CUDNN_RNN_SINGLE_REC_BIAS;
pub const CUDNN_RNN_NO_BIAS = c.CUDNN_RNN_NO_BIAS;
pub const CUDNN_FWD_MODE_INFERENCE = c.CUDNN_FWD_MODE_INFERENCE;
pub const CUDNN_FWD_MODE_TRAINING = c.CUDNN_FWD_MODE_TRAINING;
