# cuBLAS Examples

19 examples covering BLAS Level 1, 2, and 3 operations — from scalar vector ops to batched mixed-precision matrix multiply.
Enable with `-Dcublas=true`.

## Build & Run

```bash
zig build run-cublas-<name> -Dcublas=true

# Examples:
zig build run-cublas-gemm -Dcublas=true
zig build run-cublas-axpy -Dcublas=true
```

---

## Level 1 — Vector–Vector Operations

| Example | File | BLAS Op | Description |
|---------|------|---------|-------------|
| `axpy` | [axpy.zig](axpy.zig) | `SAXPY` | y = α·x + y |
| `dot` | [dot.zig](dot.zig) | `SDOT` | Dot product: Σ xᵢyᵢ |
| `nrm2_asum` | [nrm2_asum.zig](nrm2_asum.zig) | `SNRM2`, `SASUM` | L2 norm and L1 norm |
| `scal` | [scal.zig](scal.zig) | `SSCAL` | Vector scaling: x = α·x |
| `amax_amin` | [amax_amin.zig](amax_amin.zig) | `ISAMAX`, `ISAMIN` | Index of max/min absolute value |
| `swap_copy` | [swap_copy.zig](swap_copy.zig) | `SSWAP`, `SCOPY` | Vector swap and copy |
| `rot` | [rot.zig](rot.zig) | `SROT`, `SROTG` | Givens plane rotation |
| `cosine_similarity` | [cosine_similarity.zig](cosine_similarity.zig) | `SDOT`, `SNRM2` | Cosine similarity via L1 ops |

## Level 2 — Matrix–Vector Operations

| Example | File | BLAS Op | Description |
|---------|------|---------|-------------|
| `gemv` | [gemv.zig](gemv.zig) | `SGEMV` | y = α·A·x + β·y (matrix-vector) |
| `symv_syr` | [symv_syr.zig](symv_syr.zig) | `SSYMV`, `SSYR` | Symmetric matrix-vector ops |
| `trmv_trsv` | [trmv_trsv.zig](trmv_trsv.zig) | `STRMV`, `STRSV` | Triangular multiply and solve |

## Level 3 — Matrix–Matrix Operations

| Example | File | BLAS Op | Description |
|---------|------|---------|-------------|
| `gemm` | [gemm.zig](gemm.zig) | `SGEMM` | C = α·A·B + β·C (single precision) |
| `gemm_batched` | [gemm_batched.zig](gemm_batched.zig) | `SGEMMStridedBatched` | Strided batched GEMM |
| `gemm_ex` | [gemm_ex.zig](gemm_ex.zig) | `cublasGemmEx` | Mixed-precision GEMM (FP16 in, FP32 out) |
| `symm` | [symm.zig](symm.zig) | `SSYMM` | Symmetric matrix multiply: C = α·A·B + β·C |
| `trsm` | [trsm.zig](trsm.zig) | `STRSM` | Triangular solve: op(A)·X = α·B |
| `syrk` | [syrk.zig](syrk.zig) | `SSYRK` | Symmetric rank-k update: C = α·A·Aᵀ + β·C |
| `geam` | [geam.zig](geam.zig) | `SGEAM` | Matrix add / scale / transpose |
| `dgmm` | [dgmm.zig](dgmm.zig) | `SDGMM` | Diagonal matrix multiply |

---

## Row-major Note

cuBLAS uses **column-major** storage. For row-major buffers (Zig/C default), swap A↔B and M↔N in `sgemm`:

```zig
// ✅ Correct for row-major C = A × B (M×N = M×K × K×N):
try blas.sgemm(.no_transpose, .no_transpose,
    n, m, k,          // ← swap n and m
    alpha,
    b, n,             // ← B first, leading dim n
    a, k,             // ← A second, leading dim k
    beta, c, n);
```

See [`docs/cublas/README.md`](../../docs/cublas/README.md) for the full explanation.

→ Full API reference: [`docs/cublas/README.md`](../../docs/cublas/README.md)
