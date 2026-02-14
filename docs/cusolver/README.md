# cuSOLVER Module

Dense linear algebra solvers: LU, QR, SVD, Cholesky, and eigenvalue decomposition.

**Import:** `const cusolver = @import("zcuda").cusolver;`
**Enable:** `-Dcusolver=true`

## CusolverDnContext

Base context for LU and SVD operations.

```zig
fn init(ctx) !CusolverDnContext;         // Create handle
fn deinit(self) void;                    // Destroy handle
```

### LU Factorization (getrf / getrs)

```zig
fn sgetrf_bufferSize(m, n, a, lda) !i32;                           // Workspace size (float)
fn dgetrf_bufferSize(m, n, a, lda) !i32;                           // Workspace size (double)
fn sgetrf(m, n, a, lda, workspace, ipiv, info) !void;              // PA = LU (float)
fn dgetrf(m, n, a, lda, workspace, ipiv, info) !void;              // PA = LU (double)
fn sgetrs(n, nrhs, a, lda, ipiv, b, ldb, info) !void;             // Solve AX = B (float)
fn dgetrs(n, nrhs, a, lda, ipiv, b, ldb, info) !void;             // Solve AX = B (double)
```

### SVD (gesvd)

```zig
fn sgesvd_bufferSize(m, n) !i32;                                   // Workspace size (float)
fn dgesvd_bufferSize(m, n) !i32;                                   // Workspace size (double)
fn sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info) !void;  // A = UΣVᵀ (float)
fn dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info) !void;  // A = UΣVᵀ (double)
```

## CusolverDnExt

Extended context wrapping `CusolverDnContext` for Cholesky, QR, eigensolve, and Jacobi SVD.

```zig
fn init(base: *const CusolverDnContext) CusolverDnExt;
```

### Cholesky (potrf / potrs)

```zig
fn spotrf(uplo, n, a, lda, workspace, lwork, info) !void;          // A = LLᵀ (float)
fn dpotrf(uplo, n, a, lda, workspace, lwork, info) !void;          // A = LLᵀ (double)
fn spotrs(uplo, n, nrhs, a, lda, b, ldb, info) !void;             // Solve AX = B (float)
fn dpotrs(uplo, n, nrhs, a, lda, b, ldb, info) !void;             // Solve AX = B (double)
```

### QR Factorization (geqrf / orgqr)

```zig
fn sgeqrf_bufferSize(m, n, a, lda) !i32;                          // Workspace size (float)
fn dgeqrf(m, n, a, lda, tau, workspace, lwork, info) !void;       // QR (double)
fn sorgqr_bufferSize(m, n, k, a, lda, tau) !i32;                  // orgqr workspace (float)
fn dorgqr(m, n, k, a, lda, tau, workspace, lwork, info) !void;    // Extract Q (double)
```

### Eigenvalue Decomposition (syevd)

```zig
fn ssyevd_bufferSize(jobz, uplo, n, a, lda, w) !i32;              // Workspace size (float)
fn dsyevd(jobz, uplo, n, a, lda, w, workspace, lwork, info) !void; // Eigensolve (double)
```

### Jacobi SVD (gesvdj)

```zig
fn sgesvdj_bufferSize(jobz, econ, m, n, a, lda, s, u, ldu, v, ldv, params) !i32;
fn dgesvdj(jobz, econ, m, n, a, lda, s, u, ldu, v, ldv, workspace, lwork, info, params) !void;
```

## GesvdjInfo

```zig
fn init() !GesvdjInfo;                   // Create params
fn deinit(self) void;                    // Destroy params
fn setTolerance(self, tol) !void;        // Convergence tolerance
fn setMaxSweeps(self, n) !void;          // Max iterations
```

## Enums

```zig
const EigMode  = enum { no_vector, vector };
const FillMode = enum { lower, upper };
```

## Example

```zig
const cuda = @import("zcuda");

const solver = try cuda.cusolver.CusolverDnContext.init(ctx);
defer solver.deinit();

// LU factorization
const lwork = try solver.sgetrf_bufferSize(m, n, a_dev, m);
const workspace = try stream.alloc(f32, allocator, @intCast(lwork));
defer workspace.deinit();
try solver.sgetrf(m, n, a_dev, m, workspace, ipiv_dev, &info);

// SVD via extended context
const ext = cuda.cusolver.CusolverDnExt.init(&solver);
```
