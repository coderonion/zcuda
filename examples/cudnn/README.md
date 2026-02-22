# cuDNN — Deep Neural Network Examples

3 examples covering convolution, activation functions, pooling, and softmax.
Enable with `-Dcudnn=true`.

## Build & Run

```bash
zig build run-cudnn-<name> -Dcudnn=true

zig build run-cudnn-conv2d -Dcudnn=true
zig build run-cudnn-activation -Dcudnn=true
zig build run-cudnn-pooling_softmax -Dcudnn=true
```

---

## Examples

| Example | File | Description |
|---------|------|-------------|
| `conv2d` | [conv2d.zig](conv2d.zig) | 2D convolution forward pass — implicit GEMM algorithm, NCHW layout |
| `activation` | [activation.zig](activation.zig) | ReLU, sigmoid, and tanh activation functions |
| `pooling_softmax` | [pooling_softmax.zig](pooling_softmax.zig) | Max pooling → softmax pipeline |

---

## Key API

```zig
const cudnn = @import("zcuda").cudnn;

const dnn = try cudnn.CudnnContext.init(ctx);
defer dnn.deinit();

// 2D convolution (NCHW layout)
try dnn.conv2dForward(.{
    .input   = d_input,   .input_dims  = .{n, c_in, h, w},
    .filter  = d_filter,  .filter_dims = .{c_out, c_in, kh, kw},
    .output  = d_output,  .output_dims = .{n, c_out, oh, ow},
    .padding = .{pad_h, pad_w},
    .stride  = .{stride_h, stride_w},
    .algo    = .implicit_gemm,
}, stream);

// Activation
try dnn.activationForward(.relu, d_input, d_output, input_dims, stream);

// Pooling
try dnn.poolingForward(.max, .{kh, kw}, .{ph, pw}, .{sh, sw},
    d_input, input_dims, d_output, output_dims, stream);

// Softmax
try dnn.softmaxForward(.accurate, .instance, d_input, d_output, dims, stream);
```

→ Full API reference: [`docs/cudnn/README.md`](../../docs/cudnn/README.md)
