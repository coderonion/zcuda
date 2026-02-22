// src/kernel/tensor_core.zig — Tensor Core Intrinsics (WMMA / MMA / wgmma / tcgen05)
//
// All functions are SM-guarded via requireSM() at comptime.
// Fragment types represent register groups holding matrix tile data.

const arch = @import("arch.zig");
const build_options = @import("build_options");
const SM: arch.SmVersion = @enumFromInt(build_options.sm_version);

// ============================================================================
// WMMA — Warp Matrix Multiply-Accumulate (sm_70+)
// ============================================================================

/// WMMA fragment types — register arrays holding matrix tiles
pub const WmmaFragA_f16 = [8]u32;
pub const WmmaFragB_f16 = [8]u32;
pub const WmmaFragC_f16 = [4]u32;
pub const WmmaFragC_f32 = [8]f32;
pub const WmmaFragD_f16 = [4]u32;
pub const WmmaFragD_f32 = [8]f32;

/// wmma.load.a.sync — load matrix A fragment (m16n16k16, f16, row major)
pub inline fn wmma_load_a_f16(ptr: [*]const u16, stride: u32) WmmaFragA_f16 {
    arch.requireSM(SM, .sm_70, "wmma_load_a_f16()");
    var r0: u32 = undefined;
    var r1: u32 = undefined;
    var r2: u32 = undefined;
    var r3: u32 = undefined;
    var r4: u32 = undefined;
    var r5: u32 = undefined;
    var r6: u32 = undefined;
    var r7: u32 = undefined;
    asm volatile (
        \\wmma.load.a.sync.aligned.m16n16k16.row.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;
        : [o0] "=r" (r0),
          [o1] "=r" (r1),
          [o2] "=r" (r2),
          [o3] "=r" (r3),
          [o4] "=r" (r4),
          [o5] "=r" (r5),
          [o6] "=r" (r6),
          [o7] "=r" (r7),
        : [ptr] "l" (ptr),
          [stride] "r" (stride),
    );
    return .{ r0, r1, r2, r3, r4, r5, r6, r7 };
}

/// wmma.load.b.sync — load matrix B fragment (m16n16k16, f16, col major)
pub inline fn wmma_load_b_f16(ptr: [*]const u16, stride: u32) WmmaFragB_f16 {
    arch.requireSM(SM, .sm_70, "wmma_load_b_f16()");
    var r0: u32 = undefined;
    var r1: u32 = undefined;
    var r2: u32 = undefined;
    var r3: u32 = undefined;
    var r4: u32 = undefined;
    var r5: u32 = undefined;
    var r6: u32 = undefined;
    var r7: u32 = undefined;
    asm volatile (
        \\wmma.load.b.sync.aligned.m16n16k16.col.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;
        : [o0] "=r" (r0),
          [o1] "=r" (r1),
          [o2] "=r" (r2),
          [o3] "=r" (r3),
          [o4] "=r" (r4),
          [o5] "=r" (r5),
          [o6] "=r" (r6),
          [o7] "=r" (r7),
        : [ptr] "l" (ptr),
          [stride] "r" (stride),
    );
    return .{ r0, r1, r2, r3, r4, r5, r6, r7 };
}

/// wmma.load.c.sync — load accumulator C fragment (m16n16k16, f32)
pub inline fn wmma_load_c_f32(ptr: [*]const f32, stride: u32) WmmaFragC_f32 {
    arch.requireSM(SM, .sm_70, "wmma_load_c_f32()");
    var r0: f32 = undefined;
    var r1: f32 = undefined;
    var r2: f32 = undefined;
    var r3: f32 = undefined;
    var r4: f32 = undefined;
    var r5: f32 = undefined;
    var r6: f32 = undefined;
    var r7: f32 = undefined;
    asm volatile (
        \\wmma.load.c.sync.aligned.m16n16k16.row.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;
        : [o0] "=f" (r0),
          [o1] "=f" (r1),
          [o2] "=f" (r2),
          [o3] "=f" (r3),
          [o4] "=f" (r4),
          [o5] "=f" (r5),
          [o6] "=f" (r6),
          [o7] "=f" (r7),
        : [ptr] "l" (ptr),
          [stride] "r" (stride),
    );
    return .{ r0, r1, r2, r3, r4, r5, r6, r7 };
}

/// wmma.mma.sync — D = A*B + C (m16n16k16, f16 inputs, f32 accumulator)
pub inline fn wmma_mma_f16_f32(a: WmmaFragA_f16, b: WmmaFragB_f16, c: WmmaFragC_f32) WmmaFragD_f32 {
    arch.requireSM(SM, .sm_70, "wmma_mma_f16_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    var d4: f32 = undefined;
    var d5: f32 = undefined;
    var d6: f32 = undefined;
    var d7: f32 = undefined;
    asm volatile (
        \\wmma.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32
        \\  {%0,%1,%2,%3,%4,%5,%6,%7},
        \\  {%8,%9,%10,%11,%12,%13,%14,%15},
        \\  {%16,%17,%18,%19,%20,%21,%22,%23},
        \\  {%24,%25,%26,%27,%28,%29,%30,%31};
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
          [o4] "=f" (d4),
          [o5] "=f" (d5),
          [o6] "=f" (d6),
          [o7] "=f" (d7),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [a4] "r" (a[4]),
          [a5] "r" (a[5]),
          [a6] "r" (a[6]),
          [a7] "r" (a[7]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [b2] "r" (b[2]),
          [b3] "r" (b[3]),
          [b4] "r" (b[4]),
          [b5] "r" (b[5]),
          [b6] "r" (b[6]),
          [b7] "r" (b[7]),
          [c0] "f" (c[0]),
          [c1] "f" (c[1]),
          [c2] "f" (c[2]),
          [c3] "f" (c[3]),
          [c4] "f" (c[4]),
          [c5] "f" (c[5]),
          [c6] "f" (c[6]),
          [c7] "f" (c[7]),
    );
    return .{ d0, d1, d2, d3, d4, d5, d6, d7 };
}

/// wmma.mma.sync — D = A*B + C (m16n16k16, f16 inputs, f16 accumulator)
pub inline fn wmma_mma_f16_f16(a: WmmaFragA_f16, b: WmmaFragB_f16, c: WmmaFragC_f16) WmmaFragD_f16 {
    arch.requireSM(SM, .sm_70, "wmma_mma_f16_f16()");
    var d0: u32 = undefined;
    var d1: u32 = undefined;
    var d2: u32 = undefined;
    var d3: u32 = undefined;
    asm volatile (
        \\wmma.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16
        \\  {%0,%1,%2,%3},
        \\  {%4,%5,%6,%7,%8,%9,%10,%11},
        \\  {%12,%13,%14,%15,%16,%17,%18,%19},
        \\  {%20,%21,%22,%23};
        : [o0] "=r" (d0),
          [o1] "=r" (d1),
          [o2] "=r" (d2),
          [o3] "=r" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [a4] "r" (a[4]),
          [a5] "r" (a[5]),
          [a6] "r" (a[6]),
          [a7] "r" (a[7]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [b2] "r" (b[2]),
          [b3] "r" (b[3]),
          [b4] "r" (b[4]),
          [b5] "r" (b[5]),
          [b6] "r" (b[6]),
          [b7] "r" (b[7]),
          [c0] "r" (c[0]),
          [c1] "r" (c[1]),
          [c2] "r" (c[2]),
          [c3] "r" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

/// wmma.store.d.sync — store result D fragment (m16n16k16, f32, row major)
pub inline fn wmma_store_d_f32(ptr: [*]f32, d: WmmaFragD_f32, stride: u32) void {
    arch.requireSM(SM, .sm_70, "wmma_store_d_f32()");
    asm volatile (
        \\wmma.store.d.sync.aligned.m16n16k16.row.f32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7}, %9;
        :
        : [v0] "f" (d[0]),
          [v1] "f" (d[1]),
          [v2] "f" (d[2]),
          [v3] "f" (d[3]),
          [v4] "f" (d[4]),
          [v5] "f" (d[5]),
          [v6] "f" (d[6]),
          [v7] "f" (d[7]),
          [ptr] "l" (ptr),
          [stride] "r" (stride),
    );
}

// ============================================================================
// WMMA Integer — sm_75+ (Turing)
// ============================================================================

pub const WmmaFragA_s8 = [1]u32;
pub const WmmaFragB_s8 = [1]u32;
pub const WmmaFragC_s32 = [2]i32;
pub const WmmaFragD_s32 = [2]i32;

/// wmma.mma.sync — s8 (m8n8k16, s8→s32)
pub inline fn wmma_mma_s8_s32(a: WmmaFragA_s8, b: WmmaFragB_s8, c: WmmaFragC_s32) WmmaFragD_s32 {
    arch.requireSM(SM, .sm_75, "wmma_mma_s8_s32()");
    var d0: i32 = undefined;
    var d1: i32 = undefined;
    asm volatile (
        \\wmma.mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};
        : [o0] "=r" (d0),
          [o1] "=r" (d1),
        : [a0] "r" (a[0]),
          [b0] "r" (b[0]),
          [c0] "r" (c[0]),
          [c1] "r" (c[1]),
    );
    return .{ d0, d1 };
}

/// wmma.mma.sync — u8 (m8n8k16, u8→s32)
pub inline fn wmma_mma_u8_s32(a: WmmaFragA_s8, b: WmmaFragB_s8, c: WmmaFragC_s32) WmmaFragD_s32 {
    arch.requireSM(SM, .sm_75, "wmma_mma_u8_s32()");
    var d0: i32 = undefined;
    var d1: i32 = undefined;
    asm volatile (
        \\wmma.mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};
        : [o0] "=r" (d0),
          [o1] "=r" (d1),
        : [a0] "r" (a[0]),
          [b0] "r" (b[0]),
          [c0] "r" (c[0]),
          [c1] "r" (c[1]),
    );
    return .{ d0, d1 };
}

/// wmma.mma.sync — s4 (m8n8k32, s4→s32)
pub inline fn wmma_mma_s4_s32(a: WmmaFragA_s8, b: WmmaFragB_s8, c: WmmaFragC_s32) WmmaFragD_s32 {
    arch.requireSM(SM, .sm_75, "wmma_mma_s4_s32()");
    var d0: i32 = undefined;
    var d1: i32 = undefined;
    asm volatile (
        \\wmma.mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};
        : [o0] "=r" (d0),
          [o1] "=r" (d1),
        : [a0] "r" (a[0]),
          [b0] "r" (b[0]),
          [c0] "r" (c[0]),
          [c1] "r" (c[1]),
    );
    return .{ d0, d1 };
}

/// wmma.mma.sync — b1 (m8n8k128, b1→s32)
pub inline fn wmma_mma_b1_s32(a: WmmaFragA_s8, b: WmmaFragB_s8, c: WmmaFragC_s32) WmmaFragD_s32 {
    arch.requireSM(SM, .sm_75, "wmma_mma_b1_s32()");
    var d0: i32 = undefined;
    var d1: i32 = undefined;
    asm volatile (
        \\wmma.mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32 {%0,%1}, {%2}, {%3}, {%4,%5};
        : [o0] "=r" (d0),
          [o1] "=r" (d1),
        : [a0] "r" (a[0]),
          [b0] "r" (b[0]),
          [c0] "r" (c[0]),
          [c1] "r" (c[1]),
    );
    return .{ d0, d1 };
}

// ============================================================================
// MMA PTX — sm_80+ (Ampere) — lower-level than WMMA
// ============================================================================

pub const MmaFragA_f16 = [4]u32; // 8 f16 packed
pub const MmaFragB_f16 = [2]u32; // 4 f16 packed
pub const MmaFragC_f32 = [4]f32;
pub const MmaFragD_f32 = [4]f32;

/// mma.sync — f16→f32 (m16n8k16)
pub inline fn mma_f16_f32(a: MmaFragA_f16, b: MmaFragB_f16, c: MmaFragC_f32) MmaFragD_f32 {
    arch.requireSM(SM, .sm_80, "mma_f16_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        \\  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [c0] "f" (c[0]),
          [c1] "f" (c[1]),
          [c2] "f" (c[2]),
          [c3] "f" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

/// mma.sync — bf16→f32 (m16n8k16)
pub inline fn mma_bf16_f32(a: MmaFragA_f16, b: MmaFragB_f16, c: MmaFragC_f32) MmaFragD_f32 {
    arch.requireSM(SM, .sm_80, "mma_bf16_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        \\  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [c0] "f" (c[0]),
          [c1] "f" (c[1]),
          [c2] "f" (c[2]),
          [c3] "f" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

pub const MmaFragA_tf32 = [4]u32;
pub const MmaFragB_tf32 = [2]u32;

/// mma.sync — tf32→f32 (m16n8k8)
pub inline fn mma_tf32_f32(a: MmaFragA_tf32, b: MmaFragB_tf32, c: MmaFragC_f32) MmaFragD_f32 {
    arch.requireSM(SM, .sm_80, "mma_tf32_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
        \\  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [c0] "f" (c[0]),
          [c1] "f" (c[1]),
          [c2] "f" (c[2]),
          [c3] "f" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

pub const MmaFragA_f64 = [1]f64;
pub const MmaFragB_f64 = [1]f64;
pub const MmaFragC_f64 = [2]f64;
pub const MmaFragD_f64 = [2]f64;

/// mma.sync — f64→f64 (m8n8k4)
pub inline fn mma_f64_f64(a: MmaFragA_f64, b: MmaFragB_f64, c: MmaFragC_f64) MmaFragD_f64 {
    arch.requireSM(SM, .sm_80, "mma_f64_f64()");
    var d0: f64 = undefined;
    var d1: f64 = undefined;
    asm volatile (
        \\mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};
        : [o0] "=d" (d0),
          [o1] "=d" (d1),
        : [a0] "d" (a[0]),
          [b0] "d" (b[0]),
          [c0] "d" (c[0]),
          [c1] "d" (c[1]),
    );
    return .{ d0, d1 };
}

pub const MmaFragA_s8 = [2]u32;
pub const MmaFragB_s8 = [1]u32;
pub const MmaFragC_s32 = [4]i32;
pub const MmaFragD_s32 = [4]i32;

/// mma.sync — s8→s32 (m16n8k16)
pub inline fn mma_s8_s32(a: MmaFragA_s8, b: MmaFragB_s8, c: MmaFragC_s32) MmaFragD_s32 {
    arch.requireSM(SM, .sm_80, "mma_s8_s32()");
    var d0: i32 = undefined;
    var d1: i32 = undefined;
    var d2: i32 = undefined;
    var d3: i32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};
        : [o0] "=r" (d0),
          [o1] "=r" (d1),
          [o2] "=r" (d2),
          [o3] "=r" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [b0] "r" (b[0]),
          [c0] "r" (c[0]),
          [c1] "r" (c[1]),
          [c2] "r" (c[2]),
          [c3] "r" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

/// mma.sync — s8→s32 (m16n8k32)
pub inline fn mma_s8x32_s32(a: MmaFragA_f16, b: MmaFragB_f16, c: MmaFragC_s32) MmaFragD_s32 {
    arch.requireSM(SM, .sm_80, "mma_s8x32_s32()");
    var d0: i32 = undefined;
    var d1: i32 = undefined;
    var d2: i32 = undefined;
    var d3: i32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
        : [o0] "=r" (d0),
          [o1] "=r" (d1),
          [o2] "=r" (d2),
          [o3] "=r" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [c0] "r" (c[0]),
          [c1] "r" (c[1]),
          [c2] "r" (c[2]),
          [c3] "r" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

/// mma.sync — s4→s32 (m16n8k64)
pub inline fn mma_s4_s32(a: MmaFragA_f16, b: MmaFragB_f16, c: MmaFragC_s32) MmaFragD_s32 {
    arch.requireSM(SM, .sm_80, "mma_s4_s32()");
    var d0: i32 = undefined;
    var d1: i32 = undefined;
    var d2: i32 = undefined;
    var d3: i32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
        : [o0] "=r" (d0),
          [o1] "=r" (d1),
          [o2] "=r" (d2),
          [o3] "=r" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [c0] "r" (c[0]),
          [c1] "r" (c[1]),
          [c2] "r" (c[2]),
          [c3] "r" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

/// mma.sync — b1→s32 (m16n8k256)
pub inline fn mma_b1_s32(a: MmaFragA_f16, b: MmaFragB_f16, c: MmaFragC_s32) MmaFragD_s32 {
    arch.requireSM(SM, .sm_80, "mma_b1_s32()");
    var d0: i32 = undefined;
    var d1: i32 = undefined;
    var d2: i32 = undefined;
    var d3: i32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
        : [o0] "=r" (d0),
          [o1] "=r" (d1),
          [o2] "=r" (d2),
          [o3] "=r" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [c0] "r" (c[0]),
          [c1] "r" (c[1]),
          [c2] "r" (c[2]),
          [c3] "r" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

// ============================================================================
// cp.async — Asynchronous Data Pipeline (sm_80+)
// ============================================================================

/// cp.async.ca.shared.global — async copy from global to shared (4/8/16 bytes)
pub inline fn memcpy_async(dst_shared: *anyopaque, src_global: *const anyopaque, size: u32) void {
    arch.requireSM(SM, .sm_80, "memcpy_async()");
    asm volatile ("cp.async.ca.shared.global [%[dst]], [%[src]], %[sz];"
        :
        : [dst] "l" (dst_shared),
          [src] "l" (src_global),
          [sz] "r" (size),
    );
}

/// cp.async.commit_group — commit outstanding async copies as a group
pub inline fn cp_async_commit_group() void {
    arch.requireSM(SM, .sm_80, "cp_async_commit_group()");
    asm volatile ("cp.async.commit_group;");
}

/// cp.async.wait_group — wait for N most recent groups to complete
pub inline fn cp_async_wait_group(comptime n: u32) void {
    arch.requireSM(SM, .sm_80, "cp_async_wait_group()");
    if (n == 0) {
        asm volatile ("cp.async.wait_group 0;");
    } else {
        asm volatile ("cp.async.wait_group %[n];"
            :
            : [n] "n" (n),
        );
    }
}

/// cp.async.wait_all — wait for all outstanding async copies
pub inline fn cp_async_wait_all() void {
    arch.requireSM(SM, .sm_80, "cp_async_wait_all()");
    asm volatile ("cp.async.wait_all;");
}

// ============================================================================
// MMA FP8 — sm_89+ (Ada Lovelace)
// ============================================================================

pub const MmaFragA_fp8 = [4]u32;
pub const MmaFragB_fp8 = [2]u32;

/// mma.sync — e4m3→f32 (m16n8k32)
pub inline fn mma_e4m3_f32(a: MmaFragA_fp8, b: MmaFragB_fp8, c: MmaFragC_f32) MmaFragD_f32 {
    arch.requireSM(SM, .sm_89, "mma_e4m3_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
        \\  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [c0] "f" (c[0]),
          [c1] "f" (c[1]),
          [c2] "f" (c[2]),
          [c3] "f" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

/// mma.sync — e5m2→f32 (m16n8k32)
pub inline fn mma_e5m2_f32(a: MmaFragA_fp8, b: MmaFragB_fp8, c: MmaFragC_f32) MmaFragD_f32 {
    arch.requireSM(SM, .sm_89, "mma_e5m2_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
        \\  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [c0] "f" (c[0]),
          [c1] "f" (c[1]),
          [c2] "f" (c[2]),
          [c3] "f" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

/// mma.sync — mixed e4m3×e5m2→f32 (m16n8k32)
pub inline fn mma_e4m3_e5m2_f32(a: MmaFragA_fp8, b: MmaFragB_fp8, c: MmaFragC_f32) MmaFragD_f32 {
    arch.requireSM(SM, .sm_89, "mma_e4m3_e5m2_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32
        \\  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [c0] "f" (c[0]),
          [c1] "f" (c[1]),
          [c2] "f" (c[2]),
          [c3] "f" (c[3]),
    );
    return .{ d0, d1, d2, d3 };
}

// ============================================================================
// wgmma — Warp Group MMA (sm_90+ / Hopper) — 128 threads (4 warps)
// ============================================================================

pub const WgmmaFragD_f32 = [4]f32;

/// wgmma.mma_async — f16→f32 (m64n16k16)
pub inline fn wgmma_f16_f32(desc_a: u64, desc_b: u64) WgmmaFragD_f32 {
    arch.requireSM(SM, .sm_90, "wgmma_f16_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16
        \\  {%0,%1,%2,%3}, %4, %5, 1, 1, 1;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [da] "l" (desc_a),
          [db] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// wgmma.mma_async — bf16→f32
pub inline fn wgmma_bf16_f32(desc_a: u64, desc_b: u64) WgmmaFragD_f32 {
    arch.requireSM(SM, .sm_90, "wgmma_bf16_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16
        \\  {%0,%1,%2,%3}, %4, %5, 1, 1, 1;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [da] "l" (desc_a),
          [db] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// wgmma.mma_async — tf32→f32
pub inline fn wgmma_tf32_f32(desc_a: u64, desc_b: u64) WgmmaFragD_f32 {
    arch.requireSM(SM, .sm_90, "wgmma_tf32_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32
        \\  {%0,%1,%2,%3}, %4, %5, 1, 1, 1;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [da] "l" (desc_a),
          [db] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// wgmma.mma_async — e4m3→f32
pub inline fn wgmma_e4m3_f32(desc_a: u64, desc_b: u64) WgmmaFragD_f32 {
    arch.requireSM(SM, .sm_90, "wgmma_e4m3_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3
        \\  {%0,%1,%2,%3}, %4, %5, 1, 1, 1;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [da] "l" (desc_a),
          [db] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// wgmma.mma_async — e5m2→f32
pub inline fn wgmma_e5m2_f32(desc_a: u64, desc_b: u64) WgmmaFragD_f32 {
    arch.requireSM(SM, .sm_90, "wgmma_e5m2_f32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\wgmma.mma_async.sync.aligned.m64n16k32.f32.e5m2.e5m2
        \\  {%0,%1,%2,%3}, %4, %5, 1, 1, 1;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [da] "l" (desc_a),
          [db] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// wgmma.fence.sync.aligned
pub inline fn wgmma_fence() void {
    arch.requireSM(SM, .sm_90, "wgmma_fence()");
    asm volatile ("wgmma.fence.sync.aligned;");
}

/// wgmma.commit_group.sync.aligned
pub inline fn wgmma_commit_group() void {
    arch.requireSM(SM, .sm_90, "wgmma_commit_group()");
    asm volatile ("wgmma.commit_group.sync.aligned;");
}

/// wgmma.wait_group.sync.aligned
pub inline fn wgmma_wait_group(comptime n: u32) void {
    arch.requireSM(SM, .sm_90, "wgmma_wait_group()");
    if (n == 0) {
        asm volatile ("wgmma.wait_group.sync.aligned 0;");
    } else {
        asm volatile ("wgmma.wait_group.sync.aligned %[n];"
            :
            : [n] "n" (n),
        );
    }
}

// ============================================================================
// TMA — Tensor Memory Accelerator (sm_90+)
// ============================================================================

/// TMA load — async tensor copy global→shared (2D)
pub inline fn tma_load(smem_ptr: *anyopaque, desc: u64, c0: i32, c1: i32) void {
    arch.requireSM(SM, .sm_90, "tma_load()");
    asm volatile (
        \\cp.async.bulk.tensor.2d.shared::cluster.global.tile [%[dst]], [%[desc], {%[c0], %[c1]}];
        :
        : [dst] "l" (smem_ptr),
          [desc] "l" (desc),
          [c0] "r" (c0),
          [c1] "r" (c1),
    );
}

/// TMA store — async tensor copy shared→global (2D)
pub inline fn tma_store(desc: u64, smem_ptr: *const anyopaque, c0: i32, c1: i32) void {
    arch.requireSM(SM, .sm_90, "tma_store()");
    asm volatile (
        \\cp.async.bulk.tensor.2d.global.shared::cta.tile [%[desc], {%[c0], %[c1]}], [%[src]];
        :
        : [desc] "l" (desc),
          [src] "l" (smem_ptr),
          [c0] "r" (c0),
          [c1] "r" (c1),
    );
}

/// Bulk copy global→shared
pub inline fn bulk_copy_g2s(dst: *anyopaque, src: *const anyopaque, size: u32) void {
    arch.requireSM(SM, .sm_90, "bulk_copy_g2s()");
    asm volatile ("cp.async.bulk.shared::cluster.global [%[d]], [%[s]], %[sz];"
        :
        : [d] "l" (dst),
          [s] "l" (src),
          [sz] "r" (size),
    );
}

pub inline fn bulk_commit_group() void {
    arch.requireSM(SM, .sm_90, "bulk_commit_group()");
    asm volatile ("cp.async.bulk.commit_group;");
}

pub inline fn bulk_wait_group(comptime n: u32) void {
    arch.requireSM(SM, .sm_90, "bulk_wait_group()");
    if (n == 0) {
        asm volatile ("cp.async.bulk.wait_group 0;");
    } else {
        asm volatile ("cp.async.bulk.wait_group %[n];"
            :
            : [n] "n" (n),
        );
    }
}

// ============================================================================
// Cluster — Cross-SM Cooperation (sm_90+)
// ============================================================================

pub const Dim3 = @import("intrinsics.zig").Dim3;

/// cluster.sync — synchronize all blocks in the cluster
pub inline fn cluster_sync() void {
    arch.requireSM(SM, .sm_90, "cluster_sync()");
    asm volatile ("barrier.cluster.arrive;\nbarrier.cluster.wait;");
}

/// clusterDim — cluster dimensions
pub inline fn clusterDim() Dim3 {
    arch.requireSM(SM, .sm_90, "clusterDim()");
    return .{
        .x = asm volatile ("mov.u32 %[r], %nclusterid.x;"
            : [r] "=r" (-> u32),
        ),
        .y = asm volatile ("mov.u32 %[r], %nclusterid.y;"
            : [r] "=r" (-> u32),
        ),
        .z = asm volatile ("mov.u32 %[r], %nclusterid.z;"
            : [r] "=r" (-> u32),
        ),
    };
}

/// clusterIdx — current block's index within the cluster
pub inline fn clusterIdx() Dim3 {
    arch.requireSM(SM, .sm_90, "clusterIdx()");
    return .{
        .x = asm volatile ("mov.u32 %[r], %clusterid.x;"
            : [r] "=r" (-> u32),
        ),
        .y = asm volatile ("mov.u32 %[r], %clusterid.y;"
            : [r] "=r" (-> u32),
        ),
        .z = asm volatile ("mov.u32 %[r], %clusterid.z;"
            : [r] "=r" (-> u32),
        ),
    };
}

/// map_shared_cluster — map remote SM shared memory address
pub inline fn map_shared_cluster(remote_smem: *const anyopaque, rank: u32) *anyopaque {
    arch.requireSM(SM, .sm_90, "map_shared_cluster()");
    return asm volatile ("mapa.shared::cluster %[ret], %[addr], %[rank];"
        : [ret] "=l" (-> *anyopaque),
        : [addr] "l" (remote_smem),
          [rank] "r" (rank),
    );
}

/// dsmem_load — load from distributed shared memory (remote SM)
pub inline fn dsmem_load(addr: *const u32) u32 {
    arch.requireSM(SM, .sm_90, "dsmem_load()");
    return asm volatile ("ld.shared::cluster.u32 %[ret], [%[addr]];"
        : [ret] "=r" (-> u32),
        : [addr] "l" (addr),
    );
}

// ============================================================================
// tcgen05 — 5th Gen Tensor Core (sm_100+ / Blackwell)
// ============================================================================

/// tcgen05.mma — FP4
pub inline fn tcgen05_mma_fp4(desc_a: u64, desc_b: u64) [4]f32 {
    arch.requireSM(SM, .sm_100, "tcgen05_mma_fp4()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\tcgen05.mma.cta_group::1.kind::f32.fp4 {%0,%1,%2,%3}, %4, %5;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a] "l" (desc_a),
          [b] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// tcgen05.mma — FP6
pub inline fn tcgen05_mma_fp6(desc_a: u64, desc_b: u64) [4]f32 {
    arch.requireSM(SM, .sm_100, "tcgen05_mma_fp6()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\tcgen05.mma.cta_group::1.kind::f32.fp6 {%0,%1,%2,%3}, %4, %5;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a] "l" (desc_a),
          [b] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// tcgen05.mma — FP8 (e4m3)
pub inline fn tcgen05_mma_fp8(desc_a: u64, desc_b: u64) [4]f32 {
    arch.requireSM(SM, .sm_100, "tcgen05_mma_fp8()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\tcgen05.mma.cta_group::1.kind::f32.e4m3 {%0,%1,%2,%3}, %4, %5;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a] "l" (desc_a),
          [b] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// tcgen05.mma — FP16
pub inline fn tcgen05_mma_fp16(desc_a: u64, desc_b: u64) [4]f32 {
    arch.requireSM(SM, .sm_100, "tcgen05_mma_fp16()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\tcgen05.mma.cta_group::1.kind::f32.f16 {%0,%1,%2,%3}, %4, %5;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a] "l" (desc_a),
          [b] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// tcgen05.mma — BF16
pub inline fn tcgen05_mma_bf16(desc_a: u64, desc_b: u64) [4]f32 {
    arch.requireSM(SM, .sm_100, "tcgen05_mma_bf16()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\tcgen05.mma.cta_group::1.kind::f32.bf16 {%0,%1,%2,%3}, %4, %5;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a] "l" (desc_a),
          [b] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

/// tcgen05.mma — TF32
pub inline fn tcgen05_mma_tf32(desc_a: u64, desc_b: u64) [4]f32 {
    arch.requireSM(SM, .sm_100, "tcgen05_mma_tf32()");
    var d0: f32 = undefined;
    var d1: f32 = undefined;
    var d2: f32 = undefined;
    var d3: f32 = undefined;
    asm volatile (
        \\tcgen05.mma.cta_group::1.kind::f32.tf32 {%0,%1,%2,%3}, %4, %5;
        : [o0] "=f" (d0),
          [o1] "=f" (d1),
          [o2] "=f" (d2),
          [o3] "=f" (d3),
        : [a] "l" (desc_a),
          [b] "l" (desc_b),
    );
    return .{ d0, d1, d2, d3 };
}

// -- Tensor Memory Management (tcgen05) --

pub inline fn tcgen05_alloc(size: u32) u64 {
    arch.requireSM(SM, .sm_100, "tcgen05_alloc()");
    return asm volatile ("tcgen05.alloc %[ret], %[sz];"
        : [ret] "=l" (-> u64),
        : [sz] "r" (size),
    );
}

pub inline fn tcgen05_dealloc(addr: u64) void {
    arch.requireSM(SM, .sm_100, "tcgen05_dealloc()");
    asm volatile ("tcgen05.dealloc %[addr];"
        :
        : [addr] "l" (addr),
    );
}

pub inline fn tcgen05_ld(addr: u64) [4]u32 {
    arch.requireSM(SM, .sm_100, "tcgen05_ld()");
    var r0: u32 = undefined;
    var r1: u32 = undefined;
    var r2: u32 = undefined;
    var r3: u32 = undefined;
    asm volatile ("tcgen05.ld {%0,%1,%2,%3}, %4;"
        : [o0] "=r" (r0),
          [o1] "=r" (r1),
          [o2] "=r" (r2),
          [o3] "=r" (r3),
        : [addr] "l" (addr),
    );
    return .{ r0, r1, r2, r3 };
}

pub inline fn tcgen05_st(addr: u64, data: [4]u32) void {
    arch.requireSM(SM, .sm_100, "tcgen05_st()");
    asm volatile ("tcgen05.st %4, {%0,%1,%2,%3};"
        :
        : [d0] "r" (data[0]),
          [d1] "r" (data[1]),
          [d2] "r" (data[2]),
          [d3] "r" (data[3]),
          [addr] "l" (addr),
    );
}

pub inline fn tcgen05_cp(dst: u64, src: u64) void {
    arch.requireSM(SM, .sm_100, "tcgen05_cp()");
    asm volatile ("tcgen05.cp %[dst], %[src];"
        :
        : [dst] "l" (dst),
          [src] "l" (src),
    );
}

pub inline fn tcgen05_fence() void {
    arch.requireSM(SM, .sm_100, "tcgen05_fence()");
    asm volatile ("tcgen05.fence;");
}

pub inline fn tcgen05_wait() void {
    arch.requireSM(SM, .sm_100, "tcgen05_wait()");
    asm volatile ("tcgen05.wait;");
}
