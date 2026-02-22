// kernel_particle_step.zig — Euler integration step for particle system
const cuda = @import("zcuda_kernel");

export fn stepParticles(
    pos_x: [*]f32,
    pos_y: [*]f32,
    vel_x: [*]f32,
    vel_y: [*]f32,
    force_x: f32,
    force_y: f32,
    dt: f32,
    n: u32,
) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        vel_x[idx] += force_x * dt;
        vel_y[idx] += force_y * dt;
        pos_x[idx] += vel_x[idx] * dt;
        pos_y[idx] += vel_y[idx] * dt;
    }
}
