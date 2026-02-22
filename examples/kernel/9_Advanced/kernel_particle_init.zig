// kernel_particle_init.zig — Initialize particle positions/velocities
const cuda = @import("zcuda_kernel");

export fn initParticles(
    pos_x: [*]f32,
    pos_y: [*]f32,
    vel_x: [*]f32,
    vel_y: [*]f32,
    rand_data: [*]const f32,
    n: u32,
) callconv(.kernel) void {
    const idx = cuda.blockIdx().x * cuda.blockDim().x + cuda.threadIdx().x;
    if (idx < n) {
        // Initialize positions from random data, velocities to zero
        pos_x[idx] = rand_data[idx * 2] * 100.0 - 50.0;
        pos_y[idx] = rand_data[idx * 2 + 1] * 100.0 - 50.0;
        vel_x[idx] = 0.0;
        vel_y[idx] = 0.0;
    }
}
