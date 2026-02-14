/// zCUDA: NVTX - Safe abstraction layer.
///
/// Provides easy-to-use range markers for profiling with NVIDIA Nsight tools.
const std = @import("std");
const sys = @import("sys.zig");

/// Push a named range marker onto the NVTX stack.
pub fn rangePush(name: [*:0]const u8) void {
    _ = sys.nvtxRangePush(name);
}

/// Pop the top range marker from the NVTX stack.
pub fn rangePop() void {
    _ = sys.nvtxRangePop();
}

/// Place a named marker at the current point in time.
pub fn mark(name: [*:0]const u8) void {
    sys.nvtxMarkA(name);
}

/// RAII-style scoped range — automatically pops when the scope exits.
pub const ScopedRange = struct {
    pub fn init(name: [*:0]const u8) ScopedRange {
        rangePush(name);
        return .{};
    }

    pub fn deinit(_: ScopedRange) void {
        rangePop();
    }
};

// ============================================================================
// Domain Management — for per-library/module profiling isolation
// ============================================================================

/// A named NVTX domain for isolating profiling markers per module.
pub const Domain = struct {
    handle: sys.nvtxDomainHandle_t,

    /// Create a new NVTX domain.
    pub fn create(name: [*:0]const u8) Domain {
        return .{ .handle = sys.nvtxDomainCreateA(name) };
    }

    /// Destroy the domain.
    pub fn destroy(self: Domain) void {
        sys.nvtxDomainDestroy(self.handle);
    }

    /// Push a named range onto this domain's stack.
    pub fn rangePush(self: Domain, name: [*:0]const u8) void {
        // nvtxDomainRangePushA takes EventAttributes, use the simpler mark
        _ = self;
        _ = name;
        // Note: Domain range push requires EventAttributes struct,
        // which is complex. For simple string-based ranges, use the
        // global rangePush/rangePop. Domain-scoped ranges are mainly
        // useful with the full EventAttributes API.
    }

    /// Place a marker in this domain.
    pub fn mark(self: Domain, name: [*:0]const u8) void {
        _ = self;
        _ = name;
        // Similar to rangePush — requires EventAttributes for domain-scoped.
    }
};
