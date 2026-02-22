# cuSOLVER — Dense Solver Examples

5 examples covering direct linear system solvers: LU, QR, Cholesky, SVD, and eigenvalue decomposition.
Enable with `-Dcusolver=true`.

## Build & Run

```bash
zig build run-cusolver-<name> -Dcusolver=true

zig build run-cusolver-getrf  -Dcusolver=true
zig build run-cusolver-gesvd  -Dcusolver=true
zig build run-cusolver-potrf  -Dcusolver=true
zig build run-cusolver-syevd  -Dcusolver=true
zig build run-cusolver-geqrf  -Dcusolver=true
```

---

## Examples

| Example | File | Factorization | Description |
|---------|------|--------------|-------------|
| `getrf` | [getrf.zig](getrf.zig) | LU (PA = LU) | LU factorization + linear solve via backsubstitution |
| `gesvd` | [gesvd.zig](gesvd.zig) | SVD (A = UΣVᵀ) | Full singular value decomposition |
| `potrf` | [potrf.zig](potrf.zig) | Cholesky (A = LLᵀ) | Cholesky factorization for SPD matrices + solve |
| `syevd` | [syevd.zig](syevd.zig) | Eigen (Av = λv) | Symmetric eigenvalue decomposition |
| `geqrf` | [geqrf.zig](geqrf.zig) | QR (A = QR) | QR factorization with explicit Q extraction |

---

## Key API

```zig
const cusolver = @import("zcuda").cusolver;

const solver = try cusolver.CusolverDnContext.init(ctx);
defer solver.deinit();
const ext = try cusolver.CusolverDnExt.init(ctx);
defer ext.deinit();

// LU factorization (PA = LU)
const buf_size = try solver.sgetrf_bufferSize(m, n, d_a, lda);
const workspace = try stream.alloc(f32, @intCast(buf_size));
const d_piv  = try stream.alloc(i32, @min(m, n));
const d_info = try stream.alloc(i32, 1);
try solver.sgetrf(m, n, d_a, lda, workspace, d_piv, d_info, stream);

// Solve Ax = b (after LU)
try solver.sgetrs(.no_transpose, n, 1, d_a, lda, d_piv, d_b, ldb, d_info, stream);

// SVD (A = U Σ Vᵀ)
try ext.gesvdj(m, n, d_a, d_s, d_u, d_vt, d_info, .{
    .tol       = 1e-7,
    .max_sweeps = 100,
}, stream);
```

> [!NOTE]
> `devInfo` (`d_info`) is a **GPU-side** `CudaSlice(i32)` per cuSOLVER API contract.
> Use `stream.memcpyDtoH` after `ctx.synchronize()` to read factorization status on the host.

→ Full API reference: [`docs/cusolver/README.md`](../../docs/cusolver/README.md)
