/// zCUDA: NVTX - Raw FFI bindings.
const std = @import("std");
pub const c = @cImport({
    @cInclude("nvToolsExt.h");
});
pub const nvtxRangePush = c.nvtxRangePushA;
pub const nvtxRangePop = c.nvtxRangePop;
pub const nvtxMarkA = c.nvtxMarkA;

// Domain management
pub const nvtxDomainHandle_t = c.nvtxDomainHandle_t;
pub const nvtxDomainCreateA = c.nvtxDomainCreateA;
pub const nvtxDomainDestroy = c.nvtxDomainDestroy;
pub const nvtxDomainMarkA = c.nvtxDomainMarkA;
pub const nvtxDomainRangePushA = c.nvtxDomainRangePushA;
pub const nvtxDomainRangePop = c.nvtxDomainRangePop;
