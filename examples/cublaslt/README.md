# cuBLAS LT — Lightweight BLAS Examples

1 example demonstrating lightweight GEMM with algorithm heuristics.
Enable with `-Dcublaslt=true`.

## Build & Run

```bash
zig build run-cublaslt-lt_sgemm -Dcublaslt=true
```

---

## Examples

| Example | File | Description |
|---------|------|-------------|
| `lt_sgemm` | [lt_sgemm.zig](lt_sgemm.zig) | SGEMM via cuBLASLt with automatic algorithm heuristic selection |

---

## When to Use cuBLAS LT

cuBLAS LT extends cuBLAS with:

- **Algorithm search**: `getHeuristics` selects the fastest GEMM algorithm for your matrix shapes
- **Mixed precision**: FP16/BF16 input with FP32 accumulation
- **Custom epilogues**: ReLU, bias addition, GELU fused into GEMM
- **Layout control**: Row/column major, transposed, strided

## Key API

```zig
const lt = @import("zcuda").cublaslt;

const blas_lt = try lt.CublasLtContext.init(ctx);
defer blas_lt.deinit();

// Create matmul descriptor
const matmul_desc = try blas_lt.createMatmulDesc(.{
    .compute_type = .compute_32f,
    .scale_type   = .real_32f,
    .transa       = .no_transpose,
    .transb       = .no_transpose,
});

// Query heuristic
const algos = try blas_lt.getHeuristics(matmul_desc, a_layout, b_layout, c_layout, d_layout, .{});
const algo = algos[0].algo;

// Execute
try blas_lt.matmulWithAlgo(matmul_desc, alpha, d_a, a_layout, d_b, b_layout,
    beta, d_c, c_layout, d_d, d_layout, algo, workspace, stream);
```

→ Full API reference: [`docs/cublaslt/README.md`](../../docs/cublaslt/README.md)
