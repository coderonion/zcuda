# cuBLAS Module

BLAS Level 1/2/3 operations on the GPU with single and double precision.

**Import:** `const cublas = @import("zcuda").cublas;`
**Enable:** `-Dcublas=true`

## CublasContext

```zig
fn init(ctx, stream) !CublasContext;     // Create cuBLAS handle
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
const stream = ctx.defaultStream();

const cublas_ctx = try cuda.cublas.CublasContext.init(ctx, stream);
defer cublas_ctx.deinit();

// SAXPY: y = 2.0 * x + y
try cublas_ctx.saxpy(n, 2.0, x_dev, y_dev);

// SGEMM: C = A * B
try cublas_ctx.sgemm(.no_transpose, .no_transpose, m, n, k,
    1.0, a_dev, m, b_dev, k, 0.0, c_dev, m);
```
