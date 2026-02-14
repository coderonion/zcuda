/// zCUDA Unit Tests: NVTX
const cuda = @import("zcuda");
const nvtx = cuda.nvtx;

test "NVTX range push/pop" {
    nvtx.rangePush("test_range");
    nvtx.rangePop();
}

test "NVTX scoped range" {
    const range = nvtx.ScopedRange.init("scoped_test");
    defer range.deinit();
}

test "NVTX mark" {
    nvtx.mark("test_mark");
}

test "NVTX domain create and destroy" {
    const domain = nvtx.Domain.create("test_domain");
    defer domain.destroy();
}
