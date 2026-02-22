/// cuRAND Pipeline: Monte Carlo Option Pricing
///
/// Pipeline: cuRAND (generate random numbers) → custom Zig kernel (GBM path simulation)
/// Simulates stock price paths via Geometric Brownian Motion then prices European call option.
///
/// Reference: cuda-samples/randomFog, CUDALibrarySamples/cuRAND
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
const std = @import("std");
const cuda = @import("zcuda");

// kernel: gbmStep(prices, normals, r, sigma, dt, n)
//   prices: [*]f32  — in/out: current stock prices (initialized to S0, updated in place)
//   normals: [*]const f32 — N(0,1) random numbers for this step
//   r, sigma, dt: f32 — model parameters
//   n: u32 — number of paths
const kernel_gbm_paths = @import("kernel_gbm_paths");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== cuRAND Pipeline: Monte Carlo Option Pricing ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // ── Load custom kernel ──
    const mod = try kernel_gbm_paths.load(ctx, allocator);
    defer mod.deinit();
    const gbm_fn = try kernel_gbm_paths.getFunction(mod, .gbmStep);

    // ── Parameters ──
    const n_paths: u32 = 100_000; // number of simulation paths
    const n_steps: u32 = 252; // trading days in a year
    const S0: f32 = 100.0; // initial stock price
    const K: f32 = 105.0; // strike price
    const r: f32 = 0.05; // risk-free rate
    const sigma: f32 = 0.2; // volatility
    const T: f32 = 1.0; // time to maturity (years)
    const dt: f32 = T / @as(f32, @floatFromInt(n_steps));

    // ── Initialize prices to S0 on host, upload to GPU ──
    const h_prices = try allocator.alloc(f32, n_paths);
    defer allocator.free(h_prices);
    for (h_prices) |*p| p.* = S0;

    const d_prices = try stream.cloneHtoD(f32, h_prices);
    defer d_prices.deinit();

    // ── cuRAND: generate n_paths normals per step ──
    const rng = try cuda.curand.CurandContext.init(ctx, .philox4_32_10);
    defer rng.deinit();
    try rng.setStream(stream);
    try rng.setSeed(42);

    const d_normals = try stream.alloc(f32, allocator, n_paths);
    defer d_normals.deinit();

    const config = cuda.LaunchConfig.forNumElems(n_paths);

    // ── Stage: simulate n_steps GBM steps ──
    for (0..n_steps) |step| {
        // Generate fresh normal random numbers for this time step
        // fillNormal requires even-length buffer (n_paths is already even if 100_000)
        try rng.fillNormal(d_normals, 0.0, 1.0);

        // gbmStep(prices, normals, r, sigma, dt, n)
        try stream.launch(gbm_fn, config, .{
            d_prices.devicePtr(), d_normals.devicePtr(), r, sigma, dt, n_paths,
        });
        _ = step;
    }
    try stream.synchronize();
    std.debug.print("Stage 1+2: Simulated {} GBM paths × {} steps\n", .{ n_paths, n_steps });

    // ── Read back final prices ──
    const final_prices = try allocator.alloc(f32, n_paths);
    defer allocator.free(final_prices);
    try stream.memcpyDtoH(f32, final_prices, d_prices);

    // ── CPU: compute option price ──
    var payoff_sum: f64 = 0.0;
    var n_above: usize = 0;
    for (final_prices) |price| {
        const payoff: f64 = @max(0.0, @as(f64, price) - K);
        payoff_sum += payoff;
        if (price > K) n_above += 1;
    }

    const avg_payoff = payoff_sum / @as(f64, @floatFromInt(n_paths));
    const discount = @exp(-@as(f64, r) * T);
    const option_price = avg_payoff * discount;

    std.debug.print("\n── Results ({} paths) ──\n", .{n_paths});
    std.debug.print("  Paths ending above strike: {}/{}\n", .{ n_above, n_paths });
    std.debug.print("  Average payoff:  ${d:.4}\n", .{avg_payoff});
    std.debug.print("  Call option price (discounted): ${d:.4}\n", .{option_price});
    std.debug.print("  Parameters: S={d:.0}, K={d:.0}, σ={d:.0}%, r={d:.0}%, T={d:.0}y\n", .{
        @as(f64, S0), @as(f64, K), @as(f64, sigma * 100), @as(f64, r * 100), @as(f64, T),
    });

    std.debug.print("\n✓ Pipeline complete: cuRAND Normal → GBM Kernel × {} → Option Pricing\n", .{n_steps});
}
