/// Device Information Example
///
/// Prints a comprehensive GPU specification sheet:
/// 1. Device enumeration and basic properties
/// 2. Compute capability, clock rates, multiprocessor count
/// 3. Memory info (total, free, bandwidth)
/// 4. Context limits (stack size, heap, printf buffer)
/// 5. Cache configuration
///
/// Reference: cuda-samples/simpleAttributes, cudarc/12-context-config
const std = @import("std");
const cuda = @import("zcuda");

pub fn main() !void {
    std.debug.print("=== CUDA Device Information ===\n\n", .{});

    // --- Device Enumeration ---
    const device_count = try cuda.driver.CudaContext.deviceCount();
    std.debug.print("CUDA Devices Found: {}\n\n", .{device_count});

    if (device_count == 0) {
        std.debug.print("No CUDA devices available.\n", .{});
        return;
    }

    // Query each device
    for (0..@intCast(device_count)) |i| {
        const ctx = try cuda.driver.CudaContext.new(i);
        defer ctx.deinit();

        const sys = cuda.driver.sys;

        std.debug.print("━━━ Device {} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{i});
        std.debug.print("  Name:                  {s}\n", .{ctx.name()});

        const cap = try ctx.computeCapability();
        std.debug.print("  Compute Capability:    {}.{}\n", .{ cap.major, cap.minor });

        // --- Memory ---
        const total_mem = try ctx.totalMem();
        const mem_info = try ctx.memInfo();
        std.debug.print("\n  ── Memory ──\n", .{});
        std.debug.print("  Total:                 {} MB\n", .{total_mem / (1024 * 1024)});
        std.debug.print("  Free:                  {} MB\n", .{mem_info.free / (1024 * 1024)});
        std.debug.print("  Used:                  {} MB\n", .{(total_mem - mem_info.free) / (1024 * 1024)});

        const mem_bus_width = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
        const mem_clock = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
        std.debug.print("  Bus Width:             {} bit\n", .{mem_bus_width});
        std.debug.print("  Memory Clock:          {} MHz\n", .{@divTrunc(mem_clock, 1000)});
        // Bandwidth = 2 * clock * bus_width / 8 (bytes/sec)
        const bandwidth_gbps = @as(f64, @floatFromInt(mem_clock)) * 1000.0 * 2.0 *
            @as(f64, @floatFromInt(mem_bus_width)) / 8.0 / 1.0e9;
        std.debug.print("  Peak Bandwidth:        {d:.1} GB/s\n", .{bandwidth_gbps});

        // --- Compute ---
        std.debug.print("\n  ── Compute ──\n", .{});
        const sm_count = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        const clock_rate = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
        const max_threads = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        const max_block_x = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
        const max_block_y = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
        const max_block_z = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
        const max_grid_x = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
        const max_grid_y = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
        const max_grid_z = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
        const warp_size = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_WARP_SIZE);
        const regs = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);

        std.debug.print("  SMs:                   {}\n", .{sm_count});
        std.debug.print("  GPU Clock:             {} MHz\n", .{@divTrunc(clock_rate, 1000)});
        std.debug.print("  Warp Size:             {}\n", .{warp_size});
        std.debug.print("  Max Threads/Block:     {}\n", .{max_threads});
        std.debug.print("  Max Block Dim:         ({}, {}, {})\n", .{ max_block_x, max_block_y, max_block_z });
        std.debug.print("  Max Grid Dim:          ({}, {}, {})\n", .{ max_grid_x, max_grid_y, max_grid_z });
        std.debug.print("  Max Registers/Block:   {}\n", .{regs});

        const shared_mem = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
        const l2_size = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
        std.debug.print("  Shared Memory/Block:   {} KB\n", .{@divTrunc(shared_mem, 1024)});
        std.debug.print("  L2 Cache:              {} KB\n", .{@divTrunc(l2_size, 1024)});

        // --- Features ---
        std.debug.print("\n  ── Features ──\n", .{});
        const unified_addr = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
        const managed_mem = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
        const cooperative = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH);
        const concurrent_kernels = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS);
        const async_engines = try ctx.attribute(sys.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);

        std.debug.print("  Unified Addressing:    {s}\n", .{if (unified_addr != 0) "Yes" else "No"});
        std.debug.print("  Managed Memory:        {s}\n", .{if (managed_mem != 0) "Yes" else "No"});
        std.debug.print("  Cooperative Launch:    {s}\n", .{if (cooperative != 0) "Yes" else "No"});
        std.debug.print("  Concurrent Kernels:    {s}\n", .{if (concurrent_kernels != 0) "Yes" else "No"});
        std.debug.print("  Async Copy Engines:    {}\n", .{async_engines});

        // --- Context Limits ---
        std.debug.print("\n  ── Context Limits ──\n", .{});
        const stack = try ctx.getLimit(sys.CU_LIMIT_STACK_SIZE);
        const heap = try ctx.getLimit(sys.CU_LIMIT_MALLOC_HEAP_SIZE);
        const printf_buf = try ctx.getLimit(sys.CU_LIMIT_PRINTF_FIFO_SIZE);

        std.debug.print("  Stack Size/Thread:     {} bytes\n", .{stack});
        std.debug.print("  Malloc Heap:           {} MB\n", .{heap / (1024 * 1024)});
        std.debug.print("  Printf Buffer:         {} KB\n", .{printf_buf / 1024});

        // --- Cache Config ---
        const cache = try ctx.getCacheConfig();
        const cache_str: []const u8 = switch (cache) {
            0 => "No Preference", // CU_FUNC_CACHE_PREFER_NONE
            1 => "Prefer Shared", // CU_FUNC_CACHE_PREFER_SHARED
            2 => "Prefer L1", // CU_FUNC_CACHE_PREFER_L1
            3 => "Equal L1/Shared", // CU_FUNC_CACHE_PREFER_EQUAL
            else => "Unknown",
        };
        std.debug.print("  Cache Config:          {s}\n", .{cache_str});

        std.debug.print("\n", .{});
    }

    std.debug.print("✓ Device enumeration complete\n", .{});
}
