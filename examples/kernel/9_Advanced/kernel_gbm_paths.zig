// kernel_gbm_paths.zig — Geometric Brownian Motion path simulation
const cuda = @import("zcuda_kernel");

/// Apply GBM step: S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
export fn gbmStep(
    prices: [*]f32,
    normals: [*]const f32,
    r: f32,
    sigma: f32,
    dt: f32,
    n: u32,
) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        const sqrt_dt = @sqrt(dt);
        const drift = (r - 0.5 * sigma * sigma) * dt;
        const diffusion = sigma * sqrt_dt * normals[idx];
        prices[idx] = prices[idx] * cuda.__expf(drift + diffusion);
    }
}
