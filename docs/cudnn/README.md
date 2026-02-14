# cuDNN Module

Deep neural network primitives: convolution, activation, pooling, softmax, batch norm, and tensor operations.

**Import:** `const cudnn = @import("zcuda").cudnn;`
**Enable:** `-Dcudnn=true`

## CudnnContext

```zig
fn init(ctx) !CudnnContext;              // Create cuDNN handle
fn deinit(self) void;                    // Destroy handle
```

### Tensor & Filter Descriptors

```zig
fn createTensor4d(format, dtype, n, c, h, w) !TensorDescriptor;
fn createTensorNd(dtype, dims, strides) !TensorDescriptorNd;
fn createFilter4d(dtype, format, k, c, h, w) !FilterDescriptor;
fn createFilterNd(dtype, format, filter_dims) !FilterDescriptorNd;
```

### Convolution

```zig
fn createConv2d(pad_h, pad_w, str_h, str_w, dil_h, dil_w, mode, dtype) !ConvolutionDescriptor;
fn createConvNd(pads, strides, dilations, mode, dtype) !ConvolutionDescriptorNd;
fn getConvForwardWorkspaceSize(x_desc, w_desc, conv_desc, y_desc, algo) !usize;
fn getConvNdForwardOutputDim(conv_desc, input_desc, filter_desc, output_dims) !void;
fn convForward(T, α, x_desc, x, w_desc, w, conv_desc, algo, workspace, β, y_desc, y) !void;
fn convForwardNd(T, α, x_desc, x, w_desc, w, conv_desc, algo, workspace, β, y_desc, y) !void;
fn convBackwardData(T, α, w_desc, w, dy_desc, dy, conv_desc, algo, workspace, β, dx_desc, dx) !void;
fn convBackwardFilter(T, α, x_desc, x, dy_desc, dy, conv_desc, algo, workspace, β, dw_desc, dw) !void;
```

### Activation & Pooling

```zig
fn activationForward(T, act_desc, α, x_desc, x, β, y_desc, y) !void;
fn activationBackward(T, act_desc, α, y_desc, y, dy_desc, dy, x_desc, x, β, dx_desc, dx) !void;
fn poolingForward(T, pool_desc, α, x_desc, x, β, y_desc, y) !void;
fn poolingBackward(T, pool_desc, α, y_desc, y, dy_desc, dy, x_desc, x, β, dx_desc, dx) !void;
```

### Softmax

```zig
fn softmaxForward(T, algo, mode, α, x_desc, x, β, y_desc, y) !void;
fn softmaxBackward(T, algo, mode, α, y_desc, y, dy_desc, dy, β, dx_desc, dx) !void;
```

### Tensor Operations

```zig
fn opTensor(T, op_desc, α1, a_desc, a, α2, b_desc, b, β, c_desc, c) !void;
fn addTensor(T, α, a_desc, a, β, c_desc, c) !void;
fn scaleTensor(T, desc, y, α) !void;
fn getReductionWorkspaceSize(op, a_desc, c_desc) !usize;
fn reduceTensor(T, op, α, a_desc, a, β, c_desc, c, workspace) !void;
```

### Batch Normalization & Dropout

```zig
fn batchNormForward(...) !void;
fn dropoutForward(...) !void;
```

## Descriptors

| Type                      | Init Method                             | Description               |
| ------------------------- | --------------------------------------- | ------------------------- |
| `TensorDescriptor`        | `createTensor4d(...)`                   | 4D tensor (NCHW/NHWC)     |
| `TensorDescriptorNd`      | `createTensorNd(...)`                   | N-dimensional tensor      |
| `FilterDescriptor`        | `createFilter4d(...)`                   | 4D conv filter            |
| `FilterDescriptorNd`      | `createFilterNd(...)`                   | N-dimensional filter      |
| `ConvolutionDescriptor`   | `createConv2d(...)`                     | 2D convolution params     |
| `ConvolutionDescriptorNd` | `createConvNd(...)`                     | N-dimensional conv params |
| `ActivationDescriptor`    | `ActivationDescriptor.init(mode, coef)` | Activation params         |
| `PoolingDescriptor`       | `PoolingDescriptor.init2d(...)`         | 2D pooling params         |
| `OpTensorDescriptor`      | `OpTensorDescriptor.init(op, dtype)`    | Element-wise op params    |
| `DropoutDescriptor`       | —                                       | Dropout params            |

## Enums

```zig
const TensorFormat  = enum { nchw, nhwc };
const DnnDataType   = enum { f32, f64, f16 };
const ActivationMode = enum { sigmoid, relu, tanh, clipped_relu, elu, identity };
const PoolingMode   = enum { max, avg_include_padding, avg_exclude_padding, max_deterministic };
const ConvMode      = enum { convolution, cross_correlation };
const ConvFwdAlgo   = enum { implicit_gemm, implicit_precomp_gemm, gemm, direct, fft, fft_tiling, winograd, winograd_nonfused };
const SoftmaxAlgo   = enum { fast, accurate, log };
const SoftmaxMode   = enum { instance, channel };
const ReduceOp      = enum { add, mul, min, max, avg, norm1, norm2 };
const OpTensorOp    = enum { add, mul, min, max, sqrt, not };
```

## Example

```zig
const cuda = @import("zcuda");

const cudnn_ctx = try cuda.cudnn.CudnnContext.init(ctx);
defer cudnn_ctx.deinit();

// Create descriptors
const x_desc = try cudnn_ctx.createTensor4d(.nchw, .f32, 1, 3, 224, 224);
defer x_desc.deinit();

const w_desc = try cudnn_ctx.createFilter4d(.f32, .nchw, 64, 3, 3, 3);
defer w_desc.deinit();

const conv_desc = try cudnn_ctx.createConv2d(1, 1, 1, 1, 1, 1, .cross_correlation, .f32);
defer conv_desc.deinit();

// Forward convolution
try cudnn_ctx.convForward(f32, 1.0, x_desc, x, w_desc, w,
    conv_desc, .implicit_gemm, workspace, 0.0, y_desc, y);
```
