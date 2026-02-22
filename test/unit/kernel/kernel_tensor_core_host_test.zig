/// zCUDA Unit Tests: tensor_core.zig comptime validation
///
/// Tests fragment type sizes, struct declarations, function
/// existence. Runtime WMMA/MMA operations require GPU with TensorCores (sm_70+).
const std = @import("std");
const tensor_core = @import("tensor_core");

// ============================================================================
// WMMA Fragment types — comptime size validation (sm_70+)
// ============================================================================

test "WmmaFragA_f16 is [8]u32 — 32 bytes" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(tensor_core.WmmaFragA_f16));
}

test "WmmaFragB_f16 is [8]u32 — 32 bytes" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(tensor_core.WmmaFragB_f16));
}

test "WmmaFragC_f16 is [4]u32 — 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(tensor_core.WmmaFragC_f16));
}

test "WmmaFragC_f32 is [8]f32 — 32 bytes" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(tensor_core.WmmaFragC_f32));
}

test "WmmaFragD_f16 is [4]u32 — 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(tensor_core.WmmaFragD_f16));
}

test "WmmaFragD_f32 is [8]f32 — 32 bytes" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(tensor_core.WmmaFragD_f32));
}

// ============================================================================
// MMA Fragment types — comptime size validation (sm_80+)
// ============================================================================

test "MmaFragA_f16 is [4]u32 — 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(tensor_core.MmaFragA_f16));
}

test "MmaFragB_f16 is [2]u32 — 8 bytes" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(tensor_core.MmaFragB_f16));
}

test "MmaFragC_f32 is [4]f32 — 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(tensor_core.MmaFragC_f32));
}

test "MmaFragD_f32 is [4]f32 — 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(tensor_core.MmaFragD_f32));
}

// ============================================================================
// WMMA Integer Fragment types — (sm_75+)
// ============================================================================

test "WmmaFragA_s8 is [1]u32 — 4 bytes" {
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(tensor_core.WmmaFragA_s8));
}

test "WmmaFragB_s8 is [1]u32 — 4 bytes" {
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(tensor_core.WmmaFragB_s8));
}

test "WmmaFragC_s32 is [2]i32 — 8 bytes" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(tensor_core.WmmaFragC_s32));
}

test "WmmaFragD_s32 is [2]i32 — 8 bytes" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(tensor_core.WmmaFragD_s32));
}

// ============================================================================
// TF32 Fragment types — (sm_80+)
// ============================================================================

test "MmaFragA_tf32 is [4]u32 — 16 bytes" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(tensor_core.MmaFragA_tf32));
}

test "MmaFragB_tf32 is [2]u32 — 8 bytes" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(tensor_core.MmaFragB_tf32));
}

// ============================================================================
// WMMA function declarations (sm_70+)
// ============================================================================

test "tensor_core exports wmma_load_a_f16" {
    try std.testing.expect(@hasDecl(tensor_core, "wmma_load_a_f16"));
}

test "tensor_core exports wmma_load_b_f16" {
    try std.testing.expect(@hasDecl(tensor_core, "wmma_load_b_f16"));
}

test "tensor_core exports wmma_load_c_f32" {
    try std.testing.expect(@hasDecl(tensor_core, "wmma_load_c_f32"));
}

test "tensor_core exports wmma_mma_f16_f32" {
    try std.testing.expect(@hasDecl(tensor_core, "wmma_mma_f16_f32"));
}

test "tensor_core exports wmma_mma_f16_f16" {
    try std.testing.expect(@hasDecl(tensor_core, "wmma_mma_f16_f16"));
}

test "tensor_core exports wmma_store_d_f32" {
    try std.testing.expect(@hasDecl(tensor_core, "wmma_store_d_f32"));
}

// ============================================================================
// MMA function declarations (sm_80+)
// ============================================================================

test "tensor_core exports mma_f16_f32" {
    try std.testing.expect(@hasDecl(tensor_core, "mma_f16_f32"));
}

test "tensor_core exports mma_bf16_f32" {
    try std.testing.expect(@hasDecl(tensor_core, "mma_bf16_f32"));
}

test "tensor_core exports mma_tf32_f32" {
    try std.testing.expect(@hasDecl(tensor_core, "mma_tf32_f32"));
}

// ============================================================================
// Integer WMMA function declarations (sm_75+)
// ============================================================================

test "tensor_core exports wmma_mma_s8_s32" {
    try std.testing.expect(@hasDecl(tensor_core, "wmma_mma_s8_s32"));
}

test "tensor_core exports wmma_mma_u8_s32" {
    try std.testing.expect(@hasDecl(tensor_core, "wmma_mma_u8_s32"));
}

test "tensor_core exports wmma_mma_b1_s32" {
    try std.testing.expect(@hasDecl(tensor_core, "wmma_mma_b1_s32"));
}

// ============================================================================
// Fragment type element counts (via @typeInfo on the underlying array)
// ============================================================================

test "WMMA f16 A/B fragments have 8 elements each" {
    const a_info = @typeInfo(tensor_core.WmmaFragA_f16);
    const b_info = @typeInfo(tensor_core.WmmaFragB_f16);
    try std.testing.expectEqual(@as(usize, 8), a_info.array.len);
    try std.testing.expectEqual(@as(usize, 8), b_info.array.len);
}

test "WMMA f32 C/D fragments have 8 elements each" {
    const c_info = @typeInfo(tensor_core.WmmaFragC_f32);
    const d_info = @typeInfo(tensor_core.WmmaFragD_f32);
    try std.testing.expectEqual(@as(usize, 8), c_info.array.len);
    try std.testing.expectEqual(@as(usize, 8), d_info.array.len);
}

test "MMA f16 A fragment has 4 elements, B has 2" {
    const a_info = @typeInfo(tensor_core.MmaFragA_f16);
    const b_info = @typeInfo(tensor_core.MmaFragB_f16);
    try std.testing.expectEqual(@as(usize, 4), a_info.array.len);
    try std.testing.expectEqual(@as(usize, 2), b_info.array.len);
}

test "MMA f32 C/D fragments have 4 elements each" {
    const c_info = @typeInfo(tensor_core.MmaFragC_f32);
    const d_info = @typeInfo(tensor_core.MmaFragD_f32);
    try std.testing.expectEqual(@as(usize, 4), c_info.array.len);
    try std.testing.expectEqual(@as(usize, 4), d_info.array.len);
}
