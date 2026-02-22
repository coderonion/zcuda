# cuBLAS Module

BLAS Level 1/2/3 operations on the GPU with single and double precision.

**Import:** `const cublas = @import("zcuda").cublas;`
**Enable:** `-Dcublas=true`

## CublasContext

```zig
fn init(ctx) !CublasContext;             // Create cuBLAS handle
fn deinit(self) void;                    // Destroy handle
fn setStream(self, stream) !void;        // Set CUDA stream
```

## Level 1 — Vector Operations

| Method                            | Description                   | CUDA Equivalent |
| --------------------------------- | ----------------------------- | --------------- |
| `saxpy(n, α, x, y)`               | y = α·x + y (float)           | `cublasSaxpy`   |
| `daxpy(n, α, x, y)`               | y = α·x + y (double)          | `cublasDaxpy`   |
| `sscal(n, α, x)`                  | x = α·x (float)               | `cublasSscal`   |
| `dscal(n, α, x)`                  | x = α·x (double)              | `cublasDscal`   |
| `sdot(n, x, y)`                   | dot product (float)           | `cublasSdot`    |
| `ddot(n, x, y)`                   | dot product (double)          | `cublasDdot`    |
| `snrm2(n, x)`                     | L2 norm (float)               | `cublasSnrm2`   |
| `dnrm2(n, x)`                     | L2 norm (double)              | `cublasDnrm2`   |
| `sswap(n, x, y)`                  | swap x ↔ y (float)            | `cublasSswap`   |
| `dswap(n, x, y)`                  | swap x ↔ y (double)           | `cublasDswap`   |
| `scopy(n, x, y)`                  | y = x (float)                 | `cublasScopy`   |
| `dcopy(n, x, y)`                  | y = x (double)                | `cublasDcopy`   |
| `isamax(n, x)`                    | argmax(\|x\|) (float)         | `cublasIsamax`  |
| `idamax(n, x)`                    | argmax(\|x\|) (double)        | `cublasIdamax`  |
| `isamin(n, x)`                    | argmin(\|x\|) (float)         | `cublasIsamin`  |
| `idamin(n, x)`                    | argmin(\|x\|) (double)        | `cublasIdamin`  |
| `srotg(a, b, c, s)`               | Givens rotation setup (float) | `cublasSrotg`   |
| `srot(n, x, incx, y, incy, c, s)` | Apply rotation (float)        | `cublasSrot`    |
| `drot(n, x, incx, y, incy, c, s)` | Apply rotation (double)       | `cublasDrot`    |

## Level 2 — Matrix-Vector Operations

| Method                                   | Description                  | CUDA Equivalent |
| ---------------------------------------- | ---------------------------- | --------------- |
| `sgemv(trans, m, n, α, A, lda, x, β, y)` | y = α·op(A)·x + β·y (float)  | `cublasSgemv`   |
| `dgemv(trans, m, n, α, A, lda, x, β, y)` | y = α·op(A)·x + β·y (double) | `cublasDgemv`   |
| `ssymv(uplo, n, α, A, lda, x, β, y)`     | Symmetric MV (float)         | `cublasSsymv`   |
| `dsymv(uplo, n, α, A, lda, x, β, y)`     | Symmetric MV (double)        | `cublasDsymv`   |
| `strmv(uplo, trans, diag, n, A, lda, x)` | Triangular MV (float)        | `cublasStrmv`   |
| `dtrmv(uplo, trans, diag, n, A, lda, x)` | Triangular MV (double)       | `cublasDtrmv`   |
| `strsv(uplo, trans, diag, n, A, lda, x)` | Triangular solve (float)     | `cublasStrsv`   |
| `dtrsv(uplo, trans, diag, n, A, lda, x)` | Triangular solve (double)    | `cublasDtrsv`   |
| `ssyr(uplo, n, α, x, A, lda)`            | Rank-1 update (float)        | `cublasSsyr`    |
| `dsyr(uplo, n, α, x, A, lda)`            | Rank-1 update (double)       | `cublasDsyr`    |

## Level 3 — Matrix-Matrix Operations

| Method                                                   | Description                         | CUDA Equivalent              |
| -------------------------------------------------------- | ----------------------------------- | ---------------------------- |
| `sgemm(opA, opB, m, n, k, α, A, lda, B, ldb, β, C, ldc)` | C = α·A·B + β·C (float)             | `cublasSgemm`                |
| `dgemm(opA, opB, m, n, k, α, A, lda, B, ldb, β, C, ldc)` | C = α·A·B + β·C (double)            | `cublasDgemm`                |
| `sgemmStridedBatched(...)`                               | Batched GEMM (float)                | `cublasSgemmStridedBatched`  |
| `dgemmStridedBatched(...)`                               | Batched GEMM (double)               | `cublasDgemmStridedBatched`  |
| `sgemmBatched(...)`                                      | Pointer-array batched GEMM (float)  | `cublasSgemmBatched`         |
| `dgemmBatched(...)`                                      | Pointer-array batched GEMM (double) | `cublasDgemmBatched`         |
| `gemmEx(...)`                                            | Mixed-precision GEMM                | `cublasGemmEx`               |
| `gemmGroupedBatchedEx(...)`                              | Grouped batched GEMM                | `cublasGemmGroupedBatchedEx` |
| `ssymm(side, uplo, m, n, α, A, B, β, C)`                 | Symmetric MM (float)                | `cublasSsymm`                |
| `dsymm(side, uplo, m, n, α, A, B, β, C)`                 | Symmetric MM (double)               | `cublasDsymm`                |
| `strsm(side, uplo, trans, diag, m, n, α, A, B)`          | Triangular solve (float)            | `cublasStrsm`                |
| `dtrsm(side, uplo, trans, diag, m, n, α, A, B)`          | Triangular solve (double)           | `cublasDtrsm`                |
| `ssyrk(uplo, trans, n, k, α, A, β, C)`                   | Rank-k update (float)               | `cublasSsyrk`                |
| `strmm(...)`                                             | Triangular MM (float)               | `cublasStrmm`                |
| `sgeam(opA, opB, m, n, α, A, β, B, C)`                   | Matrix add (float)                  | `cublasSgeam`                |
| `dgeam(opA, opB, m, n, α, A, β, B, C)`                   | Matrix add (double)                 | `cublasDgeam`                |
| `sdgmm(side, m, n, A, x, C)`                             | Diag MM (float)                     | `cublasSdgmm`                |
| `ddgmm(side, m, n, A, x, C)`                             | Diag MM (double)                    | `cublasDdgmm`                |

## Enums

```zig
const Operation = enum { no_transpose, transpose, conj_transpose };
const FillMode  = enum { lower, upper };
const SideMode  = enum { left, right };
const DiagType  = enum { non_unit, unit };
const DataType  = enum { f16, bf16, f32, f64 };
```

## Example

```zig
const cuda = @import("zcuda");

const ctx = try cuda.driver.CudaContext.new(0);
defer ctx.deinit();

const blas = try cuda.cublas.CublasContext.init(ctx);
defer blas.deinit();

// SAXPY: y = 2.0 * x + y
try blas.saxpy(n, 2.0, x_dev, y_dev);

// SGEMM (row-major): C = A * B  — see Row-major vs Column-major below
try blas.sgemm(.no_transpose, .no_transpose, n, m, k,
    1.0, b_dev, n, a_dev, k, 0.0, c_dev, n);  // note: B before A
```

---

## ⚠️ Row-major vs Column-major

**cuBLAS always uses column-major (Fortran) storage.** Zig arrays and C arrays
are row-major by default. This mismatch is the most common source of silent
correctness bugs when using cuBLAS.

### The Problem

If you store matrices in row-major order and call cuBLAS with the naive
argument order, cuBLAS interprets the memory as a **transposed** matrix:

- Your row-major `A[i][j]` is read by cuBLAS as `A[j][i]` (column-major)
- The result `C` is the **transpose** of what you wanted

This produces large element-wise differences — not a small floating-point
error, but completely wrong values.

### The Row-major Trick (Recommended)

Exploit the mathematical identity:

```
C = A × B  (row-major)
⟺  Cᵀ = Bᵀ × Aᵀ  (column-major)
```

Since cuBLAS reads row-major data as its transpose automatically, you just
**swap the A and B arguments** (and swap M ↔ N) to get the correct result:

```zig
// Goal: C = A * B  where C is M×N, A is M×K, B is K×N  (all row-major)

// ❌ WRONG — cuBLAS produces Cᵀ when memory is read as row-major:
try blas.sgemm(.no_transpose, .no_transpose, m, n, k,
    alpha, a, m, b, k, beta, c, m);

// ✅ CORRECT — swap A↔B and m↔n:
try blas.sgemm(.no_transpose, .no_transpose, n, m, k,
    alpha, b, n, a, k, beta, c, n);
// cuBLAS computes: Cᵀ = Bᵀ × Aᵀ  ⟹  C = A × B  in row-major ✓
```

For **square** matrices (M = N = K = n), the call simplifies to just swapping
A and B while keeping all dimension arguments the same:

```zig
// Square row-major: C = A * B  (all dims = n)
try blas.sgemm(.no_transpose, .no_transpose, n, n, n,
    1.0, b, n, a, n, 0.0, c, n);  // B before A
```

### Alternative: Transpose Operations

You can also use cuBLAS transpose flags explicitly. This is equivalent but
requires different leading-dimension values:

```zig
// C = A * B  (row-major, non-square: C is M×N, A is M×K, B is K×N)
// Read A and B as transposed — tell cuBLAS they are K×M and N×K col-major:
try blas.sgemm(.transpose, .transpose, n, m, k,
    alpha, b, k, a, m, beta, c, n);
```

### Cross-validation Pattern

When validating cuBLAS against a row-major Zig GPU kernel:

```zig
// Zig kernel (row-major C = A * B):
try stream.launch(tiled_matmul_fn, cfg,
    .{ d_A.devicePtr(), d_B.devicePtr(), d_C_zig.devicePtr(), m, n, k });
try stream.synchronize();

// cuBLAS (row-major trick — swap A↔B):
try blas.sgemm(.no_transpose, .no_transpose, n, m, k,
    1.0, d_B, n, d_A, k, 0.0, d_C_blas, n);
try stream.synchronize();

// Read back and compare element-wise:
try stream.memcpyDtoH(f32, &h_zig, d_C_zig);
try stream.memcpyDtoH(f32, &h_blas, d_C_blas);

var max_diff: f32 = 0;
for (0..m * n) |i|
    max_diff = @max(max_diff, @abs(h_zig[i] - h_blas[i]));
// Expect max_diff < ~1e-2 for f32 at 256³ (exact match for integer inputs)
```

### Performance Impact

The swap trick has **zero runtime overhead** — it is a purely mathematical
reinterpretation of the same memory buffer. No extra kernel launches,
transposes, or copies occur.

| Approach | Runtime Cost | Notes |
|---|---|---|
| Swap A↔B in call | **None** | Recommended for square / regular GEMM |
| Use `.transpose` flags | **None** | More explicit, same perf |
| Explicit transpose (`sgeam`) | O(M×N) memory copy | Only if you need Aᵀ as a separate allocation |
