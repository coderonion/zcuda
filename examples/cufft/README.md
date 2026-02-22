# cuFFT — Fast Fourier Transform Examples

4 examples covering 1D, 2D, and 3D FFTs with both complex and real data.
Enable with `-Dcufft=true`.

## Build & Run

```bash
zig build run-cufft-<name> -Dcufft=true

zig build run-cufft-fft_1d_c2c -Dcufft=true
zig build run-cufft-fft_1d_r2c -Dcufft=true
zig build run-cufft-fft_2d    -Dcufft=true
zig build run-cufft-fft_3d    -Dcufft=true
```

---

## Examples

| Example | File | Transform | Description |
|---------|------|-----------|-------------|
| `fft_1d_c2c` | [fft_1d_c2c.zig](fft_1d_c2c.zig) | C2C | 1D complex-to-complex FFT + inverse round-trip |
| `fft_1d_r2c` | [fft_1d_r2c.zig](fft_1d_r2c.zig) | R2C | 1D real-to-complex FFT with frequency-domain filtering |
| `fft_2d` | [fft_2d.zig](fft_2d.zig) | C2C | 2D complex FFT on a matrix |
| `fft_3d` | [fft_3d.zig](fft_3d.zig) | C2C | 3D complex FFT on a volume |

---

## Key API

```zig
const cufft = @import("zcuda").cufft;

// 1D complex-to-complex
const plan = try cufft.CufftPlan.plan1d(n, .c2c, 1);
defer plan.deinit();

try plan.execC2C(d_input, d_output, .forward);
try plan.execC2C(d_output, d_input, .inverse); // round-trip

// 1D real-to-complex
const plan_r2c = try cufft.CufftPlan.plan1d(n, .r2c, 1);
try plan_r2c.execR2C(d_real, d_complex);

// 2D
const plan_2d = try cufft.CufftPlan.plan2d(rows, cols, .c2c);
try plan_2d.execC2C(d_in, d_out, .forward);
```

### Transform Types

| Type | Input → Output | Use Case |
|------|---------------|----------|
| `c2c` | Complex → Complex | General FFT / IFFT |
| `r2c` | Real → Complex | Real-valued signals (saves memory) |
| `c2r` | Complex → Real | Inverse of r2c |
| `z2z` | Double complex → Double complex | Double-precision FFT |

→ Full API reference: [`docs/cufft/README.md`](../../docs/cufft/README.md)
