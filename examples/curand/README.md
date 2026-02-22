# cuRAND — Random Number Generation Examples

3 examples covering GPU-accelerated random number generation with different generators and distributions.
Enable with `-Dcurand=true`.

## Build & Run

```bash
zig build run-curand-<name> -Dcurand=true

zig build run-curand-generators -Dcurand=true
zig build run-curand-distributions -Dcurand=true
zig build run-curand-monte_carlo_pi -Dcurand=true
```

---

## Examples

| Example | File | Description |
|---------|------|-------------|
| `generators` | [generators.zig](generators.zig) | Compare XORWOW, MRG32k3a, MTGP32, MT19937, Philox, and Sobol generators |
| `distributions` | [distributions.zig](distributions.zig) | Uniform, normal (Box-Muller), log-normal, and Poisson distributions |
| `monte_carlo_pi` | [monte_carlo_pi.zig](monte_carlo_pi.zig) | Monte Carlo π estimation — classic GPU RNG application |

---

## Key API

```zig
const curand = @import("zcuda").curand;

const rng = try curand.CurandContext.init(ctx, .xorwow);
defer rng.deinit();

try rng.setSeed(42);

// Generate random numbers
const d_uniform = try stream.alloc(f32, n);
try rng.generateUniform(d_uniform);           // U[0, 1)

const d_normal = try stream.alloc(f32, n);
try rng.generateNormal(d_normal, 0.0, 1.0);  // N(μ=0, σ=1)

const d_poisson = try stream.alloc(u32, n);
try rng.generatePoisson(d_poisson, 5.0);     // Poisson(λ=5)
```

### Generator Types

| Type | Description |
|------|-------------|
| `xorwow` | Default — fast, good statistical properties |
| `mrg32k3a` | Combined multiple recursive generator |
| `mtgp32` | Mersenne Twister (GPU-optimized) |
| `mt19937` | Classical Mersenne Twister |
| `philox_4x32_10` | Counter-based, highly parallel |
| `sobol32` | Quasi-random Sobol sequence |

→ Full API reference: [`docs/curand/README.md`](../../docs/curand/README.md)
