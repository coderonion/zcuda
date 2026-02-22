// examples/kernel/0_Basic/kernel_vec3_normalize.zig — Bulk Vec3 normalization
//
// Reference: cuda-samples/0_Introduction/simpleTemplates (type-generic)
// API exercised: shared.Vec3, rsqrtf, gridStrideLoop

const cuda = @import("zcuda_kernel");

/// Normalize an array of Vec3 vectors to unit length.
/// Uses fast reciprocal square root for throughput.
export fn vec3Normalize(
    vectors: [*]cuda.shared.Vec3,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        const v = vectors[i];
        const len_sq = cuda.shared.Vec3.dot(v, v);

        // Guard against zero-length vectors
        if (len_sq > 1e-12) {
            const inv_len = cuda.rsqrtf(len_sq);
            vectors[i] = cuda.shared.Vec3.scale(v, inv_len);
        }
    }
}

/// Scale all Vec3 vectors by a uniform factor
export fn vec3Scale(
    vectors: [*]cuda.shared.Vec3,
    scale: f32,
    n: u32,
) callconv(.kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        vectors[i] = cuda.shared.Vec3.scale(vectors[i], scale);
    }
}
