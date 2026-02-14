# cuBLAS LT Module

Lightweight GEMM with fine-grained control over algorithm selection and workspace.

**Import:** `const cublaslt = @import("zcuda").cublaslt;`
**Enable:** `-Dcublaslt=true`

## CublasLtContext

```zig
fn init(ctx) !CublasLtContext;                                         // Create handle
fn deinit(self) void;                                                  // Destroy handle
fn createMatmulDesc(compute, scale) !MatmulDesc;                       // Create matmul descriptor
fn createMatrixLayout(dtype, rows, cols, ld) !MatrixLayout;            // Create matrix layout
fn setTransA(self, desc, op) !void;                                    // Set transpose A
fn setTransB(self, desc, op) !void;                                    // Set transpose B
fn setLayoutBatchCount(self, layout, count) !void;                     // Set batch count
fn setLayoutStridedBatchOffset(self, layout, offset) !void;            // Set strided offset
fn getHeuristics(desc, la, lb, lc, ld, pref, results) !i32;           // Query best algorithms
fn matmul(T, desc, α, A, la, B, lb, β, C, lc, D, ld, stream) !void;  // Execute matmul (auto algo)
fn matmulWithAlgo(T, desc, α, A, la, B, lb, β, C, lc, D, ld, algo, ws, ws_size, stream) !void;  // Execute with explicit algo
```

## Free Functions

```zig
fn destroyMatmulDesc(desc) void;       // Destroy matmul descriptor
fn destroyMatrixLayout(layout) void;   // Destroy matrix layout
fn destroyPreference(pref) void;       // Destroy preference
```

## Types & Enums

```zig
const DataType    = enum { f16, bf16, f32, f64, i8, i32 };
const ComputeType = enum { f32, f64, f16, i32, f32_fast_tf32 };
const Operation   = enum { none, transpose, conjugate_transpose };
const MatmulHeuristicResult = sys.cublasLtMatmulHeuristicResult_t;
```

## Example

```zig
const cuda = @import("zcuda");

const lt = try cuda.cublaslt.CublasLtContext.init(ctx);
defer lt.deinit();

const desc = try lt.createMatmulDesc(.f32, .f32);
defer cuda.cublaslt.destroyMatmulDesc(desc);

const a_layout = try lt.createMatrixLayout(.f32, m, k, m);
const b_layout = try lt.createMatrixLayout(.f32, k, n, k);
const c_layout = try lt.createMatrixLayout(.f32, m, n, m);
defer cuda.cublaslt.destroyMatrixLayout(a_layout);
defer cuda.cublaslt.destroyMatrixLayout(b_layout);
defer cuda.cublaslt.destroyMatrixLayout(c_layout);

try lt.matmul(f32, desc, 1.0, a_dev, a_layout, b_dev, b_layout,
    0.0, c_dev, c_layout, c_dev, c_layout, stream);
```
