# cuSPARSE — Sparse Linear Algebra Examples

4 examples demonstrating sparse matrix operations with CSR, COO, and merged sparse formats.
Enable with `-Dcusparse=true`.

## Build & Run

```bash
zig build run-cusparse-<name> -Dcusparse=true

zig build run-cusparse-spmv_csr  -Dcusparse=true
zig build run-cusparse-spmv_coo  -Dcusparse=true
zig build run-cusparse-spmm_csr  -Dcusparse=true
zig build run-cusparse-spgemm    -Dcusparse=true
```

---

## Examples

| Example | File | Format | Operation |
|---------|------|--------|-----------|
| `spmv_csr` | [spmv_csr.zig](spmv_csr.zig) | CSR | Sparse matrix-vector: y = α·A·x + β·y |
| `spmv_coo` | [spmv_coo.zig](spmv_coo.zig) | COO | Sparse matrix-vector (coordinate format) |
| `spmm_csr` | [spmm_csr.zig](spmm_csr.zig) | CSR | Sparse × dense matrix multiply: C = α·A·B + β·C |
| `spgemm` | [spgemm.zig](spgemm.zig) | CSR | Sparse × sparse multiply: C = A·B |

---

## Key API

```zig
const cusparse = @import("zcuda").cusparse;

const sp = try cusparse.CusparseContext.init(ctx);
defer sp.deinit();

// Create CSR sparse matrix
const sp_mat = try sp.createCsr(
    rows, cols, nnz,
    d_row_offsets, d_col_indices, d_values,
    .i32, .i32, .general, .f32,
);

// SpMV: y = alpha * A * x + beta * y
const sp_x = try sp.createDnVec(n, d_x, .f32);
const sp_y = try sp.createDnVec(m, d_y, .f32);
try sp.spmv(.no_transpose, alpha, sp_mat, sp_x, beta, sp_y, .f32, .default, stream);

// SpGEMM: C = A * B (both sparse)
const sp_b = try sp.createCsr(rows_b, cols_b, nnz_b, ...);
const sp_c = try sp.spgemm(sp_mat, sp_b, stream);
```

### Sparse Formats

| Format | Description | Best For |
|--------|-------------|----------|
| CSR | Compressed Sparse Row | Row-major access, SpMV, SpMM |
| COO | Coordinate (row, col, val) | Incremental construction |

→ Full API reference: [`docs/cusparse/README.md`](../../docs/cusparse/README.md)
