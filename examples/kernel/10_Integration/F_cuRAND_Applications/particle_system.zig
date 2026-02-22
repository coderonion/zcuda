/// cuRAND Pipeline: Particle System Initialization
///
/// Pipeline: cuRAND (random positions + velocities) → custom Zig kernel (apply forces + constraints)
/// Demonstrates a particle system where cuRAND provides initial random state
/// and custom kernels apply physics simulation.
///
/// Reference: cuda-samples/randomFog
//
// ── Kernel Loading: Way 5 (enhanced) build.zig auto-generated bridge module ──
const std = @import("std");
const cuda = @import("zcuda");

// kernel: initParticles(pos_x, pos_y, vel_x, vel_y, rand_data, n)
// Takes separate x/y arrays for positions and velocities
const kernel_particle_init = @import("kernel_particle_init");

// kernel: stepParticles(pos_x, pos_y, vel_x, vel_y, force_x, force_y, dt, n)
// Euler integration step with separate x/y component arrays
const kernel_particle_step = @import("kernel_particle_step");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    std.debug.print("=== cuRAND Pipeline: Particle System ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    const stream = ctx.defaultStream();

    // ── Load custom kernels ──
    const mod_init = try kernel_particle_init.load(ctx, allocator);
    defer mod_init.deinit();
    const init_fn = try kernel_particle_init.getFunction(mod_init, .initParticles);

    const mod_step = try kernel_particle_step.load(ctx, allocator);
    defer mod_step.deinit();
    const step_fn = try kernel_particle_step.getFunction(mod_step, .stepParticles);

    const n_particles: u32 = 100_000;

    // ── Stage 1: cuRAND — generate 2 randoms per particle (x, y position seed) ──
    // initParticles reads rand_data[idx*2] and rand_data[idx*2+1]
    const d_randoms = try stream.alloc(f32, allocator, n_particles * 2);
    defer d_randoms.deinit();

    const rng = try cuda.curand.CurandContext.init(ctx, .philox4_32_10);
    defer rng.deinit();
    try rng.setStream(stream);
    try rng.setSeed(12345);
    try rng.fillUniform(d_randoms);
    std.debug.print("Stage 1: Generated {} random values for {} particles\n", .{ n_particles * 2, n_particles });

    // ── Stage 2: Custom kernel — transform randoms into 2D particle state ──
    // Separate x/y arrays — matching initParticles signature
    const d_pos_x = try stream.alloc(f32, allocator, n_particles);
    defer d_pos_x.deinit();
    const d_pos_y = try stream.alloc(f32, allocator, n_particles);
    defer d_pos_y.deinit();
    const d_vel_x = try stream.alloc(f32, allocator, n_particles);
    defer d_vel_x.deinit();
    const d_vel_y = try stream.alloc(f32, allocator, n_particles);
    defer d_vel_y.deinit();

    const config = cuda.LaunchConfig.forNumElems(n_particles);
    // initParticles(pos_x, pos_y, vel_x, vel_y, rand_data, n)
    try stream.launch(init_fn, config, .{
        d_pos_x.devicePtr(),   d_pos_y.devicePtr(),
        d_vel_x.devicePtr(),   d_vel_y.devicePtr(),
        d_randoms.devicePtr(), n_particles,
    });
    std.debug.print("Stage 2: Particles initialized (2D, random positions)\n", .{});

    // ── Stage 3: Custom kernel — simulate N physics steps ──
    // stepParticles(pos_x, pos_y, vel_x, vel_y, force_x, force_y, dt, n)
    const dt: f32 = 0.01;
    const gravity_x: f32 = 0.0;
    const gravity_y: f32 = -9.81;
    const n_sim_steps: u32 = 100;

    for (0..n_sim_steps) |_| {
        try stream.launch(step_fn, config, .{
            d_pos_x.devicePtr(), d_pos_y.devicePtr(),
            d_vel_x.devicePtr(), d_vel_y.devicePtr(),
            gravity_x,           gravity_y,
            dt,                  n_particles,
        });
    }
    try stream.synchronize();
    std.debug.print("Stage 3: {} physics steps simulated (dt={d:.3}s)\n", .{ n_sim_steps, dt });

    // ── Read back sample (first 5 particles) ──
    var pos_x_buf = try allocator.alloc(f32, n_particles);
    defer allocator.free(pos_x_buf);
    var pos_y_buf = try allocator.alloc(f32, n_particles);
    defer allocator.free(pos_y_buf);
    try stream.memcpyDtoH(f32, pos_x_buf, d_pos_x);
    try stream.memcpyDtoH(f32, pos_y_buf, d_pos_y);

    std.debug.print("\nFirst 5 particle positions after {} steps:\n", .{n_sim_steps});
    for (0..5) |i| {
        std.debug.print("  P{}: ({d:.3}, {d:.3})\n", .{ i, pos_x_buf[i], pos_y_buf[i] });
    }

    std.debug.print("\n✓ Pipeline complete: cuRAND → InitParticles → PhysicsStep ×{}\n", .{n_sim_steps});
}
