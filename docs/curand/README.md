# cuRAND Module

GPU random number generation with multiple distribution and generator types.

**Import:** `const curand = @import("zcuda").curand;`
**Enable:** `-Dcurand=true`

## CurandContext

```zig
fn init(ctx, rng_type) !CurandContext;                      // Create generator
fn deinit(self) void;                                       // Destroy generator
fn setSeed(self, seed) !void;                               // Set seed
fn setOffset(self, offset) !void;                           // Set generator offset
fn setDimensions(self, n) !void;                            // Set quasi-random dimensions
fn setStream(self, stream) !void;                           // Set CUDA stream
```

### Generation Functions

| Method                                    | Output Type | Description             |
| ----------------------------------------- | ----------- | ----------------------- |
| `fillUniform(data)`                       | `f32`       | Uniform (0, 1]          |
| `fillUniformDouble(data)`                 | `f64`       | Uniform (0, 1]          |
| `fillNormal(data, mean, stddev)`          | `f32`       | Normal distribution     |
| `fillNormalDouble(data, mean, stddev)`    | `f64`       | Normal distribution     |
| `fillLogNormal(data, mean, stddev)`       | `f32`       | Log-normal distribution |
| `fillLogNormalDouble(data, mean, stddev)` | `f64`       | Log-normal distribution |
| `fillUniformU32(data)`                    | `u32`       | Uniform random integers |
| `fillPoisson(data, lambda)`               | `u32`       | Poisson distribution    |

## RngType

```zig
const RngType = enum {
    default,            // Default XORWOW
    xorwow,             // XORWOW
    mrg32k3a,           // MRG32k3a combined recursive
    mtgp32,             // Mersenne Twister for GPU
    mt19937,            // Mersenne Twister 19937
    philox4_32_10,      // Philox 4×32 with 10 rounds
    sobol32,            // Sobol 32-bit quasi-random
    scrambled_sobol32,  // Scrambled Sobol 32-bit
};
```

## Example

```zig
const cuda = @import("zcuda");

const rng = try cuda.curand.CurandContext.init(ctx, .default);
defer rng.deinit();
try rng.setSeed(42);

const data = try stream.alloc(f32, allocator, 10000);
defer data.deinit();

try rng.fillUniform(data);                    // Uniform [0, 1)
try rng.fillNormal(data, 0.0, 1.0);          // Normal(0, 1)
try rng.fillPoisson(u32_data, 5.0);           // Poisson(λ=5)
```
