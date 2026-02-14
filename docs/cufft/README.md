# cuFFT Module

1D/2D/3D Fast Fourier Transform with C2C, R2C, C2R execution modes.

**Import:** `const cufft = @import("zcuda").cufft;`
**Enable:** `-Dcufft=true`

## CufftPlan

```zig
fn plan1d(nx, fft_type, batch) !CufftPlan;         // Create 1D FFT plan
fn plan2d(nx, ny, fft_type) !CufftPlan;             // Create 2D FFT plan
fn plan3d(nx, ny, nz, fft_type) !CufftPlan;         // Create 3D FFT plan
fn planMany(rank, n, ..., fft_type, batch) !CufftPlan; // Advanced batched plan
fn deinit(self) void;                                // Destroy plan
fn getSize(self) !usize;                             // Query workspace size
fn setStream(self, stream) !void;                    // Set CUDA stream
```

### Execution Functions

| Method                              | Input → Output    | Precision |
| ----------------------------------- | ----------------- | --------- |
| `execC2C(input, output, direction)` | Complex → Complex | float     |
| `execZ2Z(input, output, direction)` | Complex → Complex | double    |
| `execR2C(input, output)`            | Real → Complex    | float     |
| `execC2R(input, output)`            | Complex → Real    | float     |
| `execD2Z(input, output)`            | Real → Complex    | double    |
| `execZ2D(input, output)`            | Complex → Real    | double    |

## Enums

```zig
const FftType = enum {
    c2c_f32,  // Complex-to-complex (float)
    c2c_f64,  // Complex-to-complex (double)
    r2c_f32,  // Real-to-complex (float)
    c2r_f32,  // Complex-to-real (float)
    r2c_f64,  // Real-to-complex (double)
    c2r_f64,  // Complex-to-real (double)
};

const Direction = enum { forward, inverse };
```

## Example

```zig
const cuda = @import("zcuda");

// 1D FFT: 1024-point forward C2C
const plan = try cuda.cufft.CufftPlan.plan1d(1024, .c2c_f32, 1);
defer plan.deinit();

try plan.setStream(stream);
try plan.execC2C(input_dev, output_dev, .forward);

// 2D FFT: 256×256
const plan_2d = try cuda.cufft.CufftPlan.plan2d(256, 256, .c2c_f32);
defer plan_2d.deinit();
try plan_2d.execC2C(input_dev, output_dev, .forward);
```
