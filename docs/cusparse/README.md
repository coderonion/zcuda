# cuSPARSE Module

Sparse matrix operations: SpMV, SpMM, SpGEMM with CSR and COO formats.

**Import:** `const cusparse = @import("zcuda").cusparse;`
**Enable:** `-Dcusparse=true`

## CusparseContext

```zig
fn init(ctx) !CusparseContext;           // Create handle
fn deinit(self) void;                    // Destroy handle
```

### Sparse Matrix Creation

```zig
fn createCsr(rows, cols, nnz, row_offsets, col_indices, values) !SpMatDescr;   // CSR format
fn createCoo(rows, cols, nnz, row_indices, col_indices, values) !SpMatDescr;   // COO format
fn createCsrForSpGEMM(rows, cols, nnz, row_offsets, values) !SpMatDescr;       // CSR for SpGEMM
```

### Dense Vector/Matrix Creation

```zig
fn createDnVec(data) !DnVecDescr;                                // Dense vector
fn createDnMat(rows, cols, ld, values, dtype, order) !DnMatDescr; // Dense matrix
```

### SpMV — Sparse × Dense Vector

```zig
fn spMVBufferSize(op, α, A, x, β, y) !usize;                    // Query workspace size
fn spMV(op, α, A, x, β, y, workspace) !void;                    // y = α·op(A)·x + β·y
```

### SpMM — Sparse × Dense Matrix

```zig
fn spMMBufferSize(opA, opB, α, A, B, β, C) !usize;              // Query workspace size
fn spMM(opA, opB, α, A, B, β, C, workspace) !void;              // C = α·A·B + β·C
```

### SpGEMM — Sparse × Sparse Matrix

```zig
fn createSpGEMMDescr(self) !SpGEMMDescriptor;                     // Create SpGEMM work descriptor
fn spGEMM_workEstimation(opA, opB, α, A, B, β, C, alg, descr, buf_size, buf) !void;
fn spGEMM_compute(opA, opB, α, A, B, β, C, alg, descr, buf_size, buf) !void;
fn spGEMM_copy(opA, opB, α, A, B, β, C, alg, descr) !void;
```

### Resource Cleanup

```zig
fn destroySpMat(desc) void;              // Destroy sparse matrix
fn destroyDnVec(desc) void;              // Destroy dense vector
fn destroyDnMat(desc) void;              // Destroy dense matrix
```

## Enums

```zig
const Operation = enum { non_transpose, transpose, conjugate_transpose };
const SpGEMMAlgorithm = enum { default, csr_deterministic, csr_nondeterministic };
```

## Example

```zig
const cuda = @import("zcuda");

const sp = try cuda.cusparse.CusparseContext.init(ctx);
defer sp.deinit();

// Create CSR sparse matrix
const sp_mat = try sp.createCsr(rows, cols, nnz,
    row_offsets_dev, col_indices_dev, values_dev);
defer sp.destroySpMat(sp_mat);

// Create dense vector
const dn_x = try sp.createDnVec(x_dev);
const dn_y = try sp.createDnVec(y_dev);
defer sp.destroyDnVec(dn_x);
defer sp.destroyDnVec(dn_y);

// SpMV: y = A * x
const buf_size = try sp.spMVBufferSize(.non_transpose, 1.0, sp_mat, dn_x, 0.0, dn_y);
const workspace = try stream.alloc(u8, allocator, buf_size);
defer workspace.deinit();
try sp.spMV(.non_transpose, 1.0, sp_mat, dn_x, 0.0, dn_y, workspace);
```
