/// Cosine Similarity Example
///
/// Computes cosine similarity between two vectors using cuBLAS Level-1 operations:
///   cos(θ) = (a · b) / (||a||₂ × ||b||₂)
///
/// Uses: sdot (dot product) + snrm2 (L2 norm)
/// Real use case: comparing text embedding vectors (768-dim, BERT-like)
///
/// Reference: Composite of CUDALibrarySamples Level-1/dot + Level-1/nrm2
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== Cosine Similarity Example ===\n\n", .{});

    const ctx = try cuda.driver.CudaContext.new(0);
    defer ctx.deinit();

    const stream = ctx.defaultStream();
    const blas = try cuda.cublas.CublasContext.init(ctx);
    defer blas.deinit();

    // Simulate 768-dimensional "text embeddings"
    const dim: usize = 768;
    const n: i32 = @intCast(dim);

    var emb_a: [dim]f32 = undefined;
    var emb_b: [dim]f32 = undefined;
    var emb_c: [dim]f32 = undefined;

    // emb_a and emb_b: similar vectors (simulating similar sentences)
    // emb_c: quite different vector (simulating unrelated sentence)
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (&emb_a, &emb_b, &emb_c) |*a, *b, *c| {
        const base = random.float(f32) * 2.0 - 1.0;
        a.* = base;
        b.* = base + (random.float(f32) - 0.5) * 0.2; // Close to a
        c.* = random.float(f32) * 2.0 - 1.0; // Independent
    }

    // Copy to device
    const d_a = try stream.cloneHtod(f32, &emb_a);
    defer d_a.deinit();
    const d_b = try stream.cloneHtod(f32, &emb_b);
    defer d_b.deinit();
    const d_c = try stream.cloneHtod(f32, &emb_c);
    defer d_c.deinit();

    // Compute norms
    const norm_a = try blas.snrm2(n, d_a);
    const norm_b = try blas.snrm2(n, d_b);
    const norm_c = try blas.snrm2(n, d_c);

    std.debug.print("Embedding dimensions: {}\n", .{dim});
    std.debug.print("  ||emb_A||₂ = {d:.4}\n", .{norm_a});
    std.debug.print("  ||emb_B||₂ = {d:.4}\n", .{norm_b});
    std.debug.print("  ||emb_C||₂ = {d:.4}\n\n", .{norm_c});

    // Compute cosine similarities
    const dot_ab = try blas.sdot(n, d_a, d_b);
    const cos_ab = dot_ab / (norm_a * norm_b);

    const dot_ac = try blas.sdot(n, d_a, d_c);
    const cos_ac = dot_ac / (norm_a * norm_c);

    const dot_bc = try blas.sdot(n, d_b, d_c);
    const cos_bc = dot_bc / (norm_b * norm_c);

    std.debug.print("─── Cosine Similarities ───\n", .{});
    std.debug.print("  cos(A, B) = {d:.6}  (similar sentences)\n", .{cos_ab});
    std.debug.print("  cos(A, C) = {d:.6}  (unrelated)\n", .{cos_ac});
    std.debug.print("  cos(B, C) = {d:.6}  (unrelated)\n\n", .{cos_bc});

    // The similar vectors should have higher cosine similarity
    std.debug.print("─── Interpretation ───\n", .{});
    if (cos_ab > cos_ac and cos_ab > cos_bc) {
        std.debug.print("  ✓ Sentences A and B are most similar (expected)\n", .{});
    } else {
        std.debug.print("  ⚠ Unexpected similarity ranking\n", .{});
    }
    std.debug.print("  cos(A,B)={d:.4} > cos(A,C)={d:.4}, cos(B,C)={d:.4}\n", .{ cos_ab, cos_ac, cos_bc });

    std.debug.print("\n✓ Cosine similarity example complete\n", .{});
}
