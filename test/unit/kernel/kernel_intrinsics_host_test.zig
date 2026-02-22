/// zCUDA Unit Tests: intrinsics.zig type signatures and metadata
///
/// Tests comptime-visible properties of intrinsics.zig:
/// - Dim3 struct layout
/// - warpSize and FULL_MASK constants
/// - Function declaration existence (threadIdx, blockIdx, atomics, fast math, etc.)
///
/// Runtime behavior requires GPU (nvptx64 target), so only declarations
/// and constants are tested here.
const std = @import("std");
const intrinsics = @import("intrinsics");

// ============================================================================
// Dim3 — struct layout
// ============================================================================

test "Dim3 has x, y, z fields" {
    const info = @typeInfo(intrinsics.Dim3);
    try std.testing.expect(info == .@"struct");
    const fields = info.@"struct".fields;
    try std.testing.expectEqual(@as(usize, 3), fields.len);
    try std.testing.expectEqualStrings("x", fields[0].name);
    try std.testing.expectEqualStrings("y", fields[1].name);
    try std.testing.expectEqualStrings("z", fields[2].name);
}

test "Dim3 size is 12 bytes (3 × u32)" {
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(intrinsics.Dim3));
}

// ============================================================================
// Constants
// ============================================================================

test "warpSize is 32" {
    try std.testing.expectEqual(@as(u32, 32), intrinsics.warpSize);
}

test "FULL_MASK is 0xffffffff" {
    try std.testing.expectEqual(@as(u32, 0xffffffff), intrinsics.FULL_MASK);
}

// ============================================================================
// Thread indexing function declarations
// ============================================================================

test "intrinsics exports threadIdx" {
    try std.testing.expect(@hasDecl(intrinsics, "threadIdx"));
}

test "intrinsics exports blockIdx" {
    try std.testing.expect(@hasDecl(intrinsics, "blockIdx"));
}

test "intrinsics exports blockDim" {
    try std.testing.expect(@hasDecl(intrinsics, "blockDim"));
}

test "intrinsics exports gridDim" {
    try std.testing.expect(@hasDecl(intrinsics, "gridDim"));
}

// ============================================================================
// Synchronization function declarations
// ============================================================================

test "intrinsics exports __syncthreads" {
    try std.testing.expect(@hasDecl(intrinsics, "__syncthreads"));
}

test "intrinsics exports __threadfence" {
    try std.testing.expect(@hasDecl(intrinsics, "__threadfence"));
}

test "intrinsics exports __threadfence_block" {
    try std.testing.expect(@hasDecl(intrinsics, "__threadfence_block"));
}

test "intrinsics exports __threadfence_system" {
    try std.testing.expect(@hasDecl(intrinsics, "__threadfence_system"));
}

// ============================================================================
// Atomic function declarations
// ============================================================================

test "intrinsics exports atomicAdd" {
    try std.testing.expect(@hasDecl(intrinsics, "atomicAdd"));
}

test "intrinsics exports atomicCAS" {
    try std.testing.expect(@hasDecl(intrinsics, "atomicCAS"));
}

test "intrinsics exports atomicExch" {
    try std.testing.expect(@hasDecl(intrinsics, "atomicExch"));
}

test "intrinsics exports atomicMin" {
    try std.testing.expect(@hasDecl(intrinsics, "atomicMin"));
}

test "intrinsics exports atomicMax" {
    try std.testing.expect(@hasDecl(intrinsics, "atomicMax"));
}

test "intrinsics exports atomicAnd" {
    try std.testing.expect(@hasDecl(intrinsics, "atomicAnd"));
}

test "intrinsics exports atomicOr" {
    try std.testing.expect(@hasDecl(intrinsics, "atomicOr"));
}

// ============================================================================
// Warp shuffle function declarations
// ============================================================================

test "intrinsics exports __shfl_sync" {
    try std.testing.expect(@hasDecl(intrinsics, "__shfl_sync"));
}

test "intrinsics exports __shfl_down_sync" {
    try std.testing.expect(@hasDecl(intrinsics, "__shfl_down_sync"));
}

test "intrinsics exports __shfl_up_sync" {
    try std.testing.expect(@hasDecl(intrinsics, "__shfl_up_sync"));
}

test "intrinsics exports __shfl_xor_sync" {
    try std.testing.expect(@hasDecl(intrinsics, "__shfl_xor_sync"));
}

test "intrinsics exports __ballot_sync" {
    try std.testing.expect(@hasDecl(intrinsics, "__ballot_sync"));
}

// ============================================================================
// Fast math function declarations
// ============================================================================

test "intrinsics exports __sinf" {
    try std.testing.expect(@hasDecl(intrinsics, "__sinf"));
}

test "intrinsics exports __cosf" {
    try std.testing.expect(@hasDecl(intrinsics, "__cosf"));
}

test "intrinsics exports __expf" {
    try std.testing.expect(@hasDecl(intrinsics, "__expf"));
}

test "intrinsics exports __logf" {
    try std.testing.expect(@hasDecl(intrinsics, "__logf"));
}

test "intrinsics exports __log2f" {
    try std.testing.expect(@hasDecl(intrinsics, "__log2f"));
}

test "intrinsics exports __exp2f" {
    try std.testing.expect(@hasDecl(intrinsics, "__exp2f"));
}

test "intrinsics exports sqrtf" {
    try std.testing.expect(@hasDecl(intrinsics, "sqrtf"));
}

test "intrinsics exports rsqrtf" {
    try std.testing.expect(@hasDecl(intrinsics, "rsqrtf"));
}

test "intrinsics exports fabsf" {
    try std.testing.expect(@hasDecl(intrinsics, "fabsf"));
}

test "intrinsics exports fminf" {
    try std.testing.expect(@hasDecl(intrinsics, "fminf"));
}

test "intrinsics exports fmaxf" {
    try std.testing.expect(@hasDecl(intrinsics, "fmaxf"));
}

test "intrinsics exports __fmaf_rn" {
    try std.testing.expect(@hasDecl(intrinsics, "__fmaf_rn"));
}

test "intrinsics exports __fdividef" {
    try std.testing.expect(@hasDecl(intrinsics, "__fdividef"));
}

// ============================================================================
// Clock function declarations
// ============================================================================

test "intrinsics exports clock" {
    try std.testing.expect(@hasDecl(intrinsics, "clock"));
}

test "intrinsics exports clock64" {
    try std.testing.expect(@hasDecl(intrinsics, "clock64"));
}

// ============================================================================
// Warp vote function declarations
// ============================================================================

test "intrinsics exports __all_sync" {
    try std.testing.expect(@hasDecl(intrinsics, "__all_sync"));
}

test "intrinsics exports __any_sync" {
    try std.testing.expect(@hasDecl(intrinsics, "__any_sync"));
}

// ============================================================================
// Special intrinsics
// ============================================================================

test "intrinsics exports __popc" {
    try std.testing.expect(@hasDecl(intrinsics, "__popc"));
}

test "intrinsics exports __clz" {
    try std.testing.expect(@hasDecl(intrinsics, "__clz"));
}

test "intrinsics exports __ffs" {
    try std.testing.expect(@hasDecl(intrinsics, "__ffs"));
}

test "intrinsics exports __brev" {
    try std.testing.expect(@hasDecl(intrinsics, "__brev"));
}
