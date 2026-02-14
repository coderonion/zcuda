/// Struct Kernel Example
///
/// Demonstrates passing Zig `extern struct` to GPU kernels.
/// Key concepts:
/// 1. Define matching struct layouts in Zig and CUDA C++
/// 2. Pass struct by value as a kernel argument
/// 3. Zig `extern struct` ensures C-compatible memory layout
///
/// Reference: cudarc/05-device-repr
const std = @import("std");
const cuda = @import("zcuda");

/// A particle struct with C-compatible layout.
/// The `extern` keyword ensures Zig uses C struct alignment/padding rules,
/// matching the CUDA kernel's struct layout exactly.
const Particle = extern struct {
    x: f32,
    y: f32,
    z: f32,
    mass: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    charge: f32,
};

const kernel_src =
    \\struct Particle {
    \\    float x, y, z;
    \\    float mass;
    \\    float vx, vy, vz;
    \\    float charge;
    \\};
    \\
    \\extern "C" __global__ void apply_gravity(
    \\    Particle *particles,
    \\    float dt,
    \\    float gravity,
    \\    int n
    \\) {
    \\    int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        // Apply gravity to vertical velocity
    \\        particles[i].vy += gravity * dt;
    \\        // Update positions based on velocity
    \\        particles[i].x += particles[i].vx * dt;
    \\        particles[i].y += particles[i].vy * dt;
    \\        particles[i].z += particles[i].vz * dt;
    \\    }
    \\}
;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Struct Kernel Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();
    std.debug.print("Device: {s}\n", .{ctx.name()});
    std.debug.print("Particle struct size: {} bytes\n\n", .{@sizeOf(Particle)});

    const stream = ctx.defaultStream();

    // Compile kernel
    const ptx = try cuda.nvrtc.compilePtx(allocator, kernel_src);
    defer allocator.free(ptx);
    const module = try ctx.loadModule(ptx);
    defer module.deinit();
    const kernel = try module.getFunction("apply_gravity");

    // --- Create particles ---
    const n: usize = 5;
    const n_i32: i32 = @intCast(n);
    const dt: f32 = 0.016; // ~60 FPS time step
    const gravity: f32 = -9.81;

    var particles: [5]Particle = .{
        .{ .x = 0.0, .y = 10.0, .z = 0.0, .mass = 1.0, .vx = 5.0, .vy = 0.0, .vz = 0.0, .charge = 1.0 },
        .{ .x = 1.0, .y = 20.0, .z = 0.0, .mass = 2.0, .vx = 0.0, .vy = 3.0, .vz = 0.0, .charge = -1.0 },
        .{ .x = -3.0, .y = 5.0, .z = 2.0, .mass = 0.5, .vx = 1.0, .vy = 1.0, .vz = -1.0, .charge = 0.5 },
        .{ .x = 0.0, .y = 0.0, .z = 0.0, .mass = 10.0, .vx = 0.0, .vy = 20.0, .vz = 0.0, .charge = 0.0 },
        .{ .x = 5.0, .y = 15.0, .z = -1.0, .mass = 3.0, .vx = -2.0, .vy = -1.0, .vz = 3.0, .charge = 2.0 },
    };

    std.debug.print("─── Before Simulation ───\n", .{});
    printParticles(&particles);

    // Copy particles to device
    const d_particles = try stream.cloneHtod(Particle, &particles);
    defer d_particles.deinit();

    // Simulate 10 time steps
    const steps: usize = 10;
    const config = cuda.LaunchConfig.forNumElems(@intCast(n));

    for (0..steps) |_| {
        try stream.launch(kernel, config, .{ &d_particles, dt, gravity, n_i32 });
    }
    try stream.synchronize();

    // Copy results back
    try stream.memcpyDtoh(Particle, &particles, d_particles);

    std.debug.print("\n─── After {} Steps (dt={d:.3}s, g={d:.2}) ───\n", .{ steps, dt, gravity });
    printParticles(&particles);

    // Verify: particle 3 was launched straight up (vy=20), should still be above ground
    const p3 = particles[3];
    const time_elapsed = dt * @as(f32, @floatFromInt(steps));
    const expected_y = 20.0 * time_elapsed + 0.5 * gravity * time_elapsed * time_elapsed;
    std.debug.print("\nVerification (Particle 3, launched upward):\n", .{});
    std.debug.print("  Time elapsed: {d:.3}s\n", .{time_elapsed});
    std.debug.print("  GPU y={d:.4}, analytical y≈{d:.4}\n", .{ p3.y, expected_y });

    std.debug.print("\n✓ Struct kernel example complete\n", .{});
}

fn printParticles(particles: []const Particle) void {
    std.debug.print("{s:>4}  {s:>8}  {s:>8}  {s:>8}  {s:>6}  {s:>8}  {s:>8}  {s:>8}\n", .{
        "#", "x", "y", "z", "mass", "vx", "vy", "vz",
    });
    for (particles, 0..) |p, i| {
        std.debug.print("{:4}  {d:8.3}  {d:8.3}  {d:8.3}  {d:6.1}  {d:8.3}  {d:8.3}  {d:8.3}\n", .{
            i, p.x, p.y, p.z, p.mass, p.vx, p.vy, p.vz,
        });
    }
}
