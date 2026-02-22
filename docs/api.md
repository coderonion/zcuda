# zCUDA API Reference

Complete API reference for zCUDA's safe abstraction layer.
zCUDA follows a **three-layer architecture** for each CUDA library:

| Layer      | Module              | Purpose                                            |
| ---------- | ------------------- | -------------------------------------------------- |
| **sys**    | `cuda.<lib>.sys`    | Raw C bindings (auto-generated from CUDA headers)  |
| **result** | `cuda.<lib>.result` | Zig-idiomatic wrappers that return `error!T`       |
| **safe**   | `cuda.<lib>.*`      | High-level, type-safe API with resource management |

User code should exclusively use the **safe layer** (top-level module exports).

> For detailed per-module documentation with examples, see the [module docs](README.md).

---

## Table of Contents

- [Driver API](#driver-api) — [full docs](driver/README.md)
- [cuBLAS](#cublas) — [full docs](cublas/README.md)
- [cuBLAS LT](#cublas-lt) — [full docs](cublaslt/README.md)
- [cuDNN](#cudnn) — [full docs](cudnn/README.md)
- [cuFFT](#cufft) — [full docs](cufft/README.md)
- [cuRAND](#curand) — [full docs](curand/README.md)
- [cuSOLVER](#cusolver) — [full docs](cusolver/README.md)
- [cuSPARSE](#cusparse) — [full docs](cusparse/README.md)
- [NVRTC](#nvrtc) — [full docs](nvrtc/README.md)
- [NVTX](#nvtx) — [full docs](nvtx/README.md)
- [Kernel DSL](#kernel-dsl) — [full docs](kernel/API.md)

---

## Driver API

**Import:** `const driver = @import("zcuda").driver;`
📖 [Full documentation](driver/README.md)

### CudaContext

GPU context management with automatic CUDA driver initialization.

| Method              | Signature                                         | Description                 |
| ------------------- | ------------------------------------------------- | --------------------------- |
| `new`               | `fn new(ordinal: usize) !*CudaContext`            | Create context for device N |
| `deinit`            | `fn deinit(self)`                                 | Destroy context             |
| `deviceCount`       | `fn deviceCount() !i32`                           | Number of CUDA devices      |
| `name`              | `fn name(self) []const u8`                        | Device name string          |
| `computeCapability` | `fn computeCapability(self) !struct{major,minor}` | SM version                  |
| `totalMem`          | `fn totalMem(self) !usize`                        | Total device memory (bytes) |
| `memInfo`           | `fn memInfo(self) !struct{free,total}`            | Free/total memory           |
| `attribute`         | `fn attribute(self, attr) !i32`                   | Query device attribute      |
| `getLimit`          | `fn getLimit(self, limit) !usize`                 | Query context limit         |
| `setLimit`          | `fn setLimit(self, limit, value) !void`           | Set context limit           |
| `getCacheConfig`    | `fn getCacheConfig(self) !CUfunc_cache`           | L1/shared preference        |
| `setCacheConfig`    | `fn setCacheConfig(self, config) !void`           | Set L1/shared preference    |
| `synchronize`       | `fn synchronize(self) !void`                      | Synchronize context         |
| `defaultStream`     | `fn defaultStream(self) *const CudaStream`        | Default stream              |
| `newStream`         | `fn newStream(self) !CudaStream`                  | Create non-blocking stream  |
| `loadModule`        | `fn loadModule(self, ptx) !CudaModule`            | Load PTX module             |
| `createEvent`       | `fn createEvent(self, flags) !CudaEvent`          | Create event                |
| `allocManaged`      | `fn allocManaged(self, T, len) !CudaSlice(T)`     | Unified memory              |

### CudaStream

| Method                                   | Description                    |
| ---------------------------------------- | ------------------------------ |
| `alloc(T, alloc, n) !CudaSlice(T)`       | Allocate device memory         |
| `allocZeros(T, alloc, n) !CudaSlice(T)`  | Alloc + zero-fill on device    |
| `cloneHtoD(T, host_slice) !CudaSlice(T)` | Host → Device copy             |
| `memcpyHtoD(T, dst, src) !void`          | Copy host → device             |
| `memcpyDtoH(T, dst, src) !void`          | Copy device → host             |
| `cloneDtoH(T, alloc, src) ![]T`          | Clone device → new host buffer |
| `memcpyDtoD(T, dst, src) !void`          | Copy device → device           |
| `memcpyHtoDAsync(T, dst, src) !void`     | Async host → device            |
| `memcpyDtoHAsync(T, dst, src) !void`     | Async device → host            |
| `memcpyDtoDAsync(T, dst, src) !void`     | Async device → device          |
| `launch(kernel, config, args) !void`     | Launch kernel                  |
| `synchronize(self) !void`                | Synchronize stream             |
| `waitEvent(self, event) !void`           | Wait for event                 |
| `query(self) !bool`                      | Non-blocking completion check  |
| `createEvent(flags) !CudaEvent`          | Create event                   |
| `recordEvent(event) !void`               | Record event                   |
| `prefetchAsync(T, slice) !void`          | Prefetch unified memory        |
| `beginCapture(self) !void`               | Begin graph capture            |
| `endCapture(self) !?CudaGraph`           | End capture → executable graph |

### CudaSlice(T) / CudaView(T) / CudaViewMut(T)

| Type             | Method                                | Description        |
| ---------------- | ------------------------------------- | ------------------ |
| `CudaSlice(T)`   | `deinit(self)`                        | Free device memory |
| `CudaSlice(T)`   | `slice(start, end) CudaView(T)`       | Immutable sub-view |
| `CudaSlice(T)`   | `sliceMut(start, end) CudaViewMut(T)` | Mutable sub-view   |
| `CudaSlice(T)`   | `devicePtr(self) DevicePtr(T)`        | Get typed pointer  |
| `CudaView(T)`    | `subView(start, end) CudaView(T)`     | Sub-view           |
| `CudaViewMut(T)` | `subView(start, end) CudaViewMut(T)`  | Mutable sub-view   |

### CudaEvent

| Method                         | Description                   |
| ------------------------------ | ----------------------------- |
| `record(self, stream) !void`   | Record event                  |
| `synchronize(self) !void`      | Wait for event                |
| `elapsedTime(start, end) !f32` | Milliseconds between events   |
| `query(self) !bool`            | Non-blocking completion check |
| `deinit(self)`                 | Destroy event                 |

### CudaModule / CudaFunction

| Type           | Method                            | Description              |
| -------------- | --------------------------------- | ------------------------ |
| `CudaModule`   | `getFunction(name) !CudaFunction` | Get kernel by name       |
| `CudaModule`   | `deinit(self)`                    | Unload module            |
| `CudaFunction` | `getAttribute(attrib) !i32`       | Query function attribute |

### CudaGraph

| Method               | Description           |
| -------------------- | --------------------- |
| `launch(self) !void` | Replay recorded graph |
| `deinit(self)`       | Destroy graph         |

---

## cuBLAS

**Import:** `const cublas = @import("zcuda").cublas;`
**Enable:** `-Dcublas=true`
📖 [Full documentation](cublas/README.md)

### CublasContext

| Method                             | Description          |
| ---------------------------------- | -------------------- |
| `init(ctx, stream) !CublasContext` | Create cuBLAS handle |
| `deinit(self)`                     | Destroy handle       |
| `setStream(self, stream) !void`    | Set CUDA stream      |

### Level 1 (Vector)

| Method            | Description           | CUDA Equivalent                 |
| ----------------- | --------------------- | ------------------------------- |
| `saxpy / daxpy`   | y = α·x + y           | `cublasSaxpy` / `cublasDaxpy`   |
| `sscal / dscal`   | x = α·x               | `cublasSscal` / `cublasDscal`   |
| `sdot / ddot`     | dot product           | `cublasSdot` / `cublasDdot`     |
| `snrm2 / dnrm2`   | L2 norm               | `cublasSnrm2` / `cublasDnrm2`   |
| `isamax / idamax` | argmax(\|x\|)         | `cublasIsamax` / `cublasIdamax` |
| `isamin / idamin` | argmin(\|x\|)         | `cublasIsamin` / `cublasIdamin` |
| `sswap / dswap`   | swap x ↔ y            | `cublasSswap` / `cublasDswap`   |
| `scopy / dcopy`   | y = x                 | `cublasScopy` / `cublasDcopy`   |
| `srotg`           | Givens rotation setup | `cublasSrotg`                   |
| `srot / drot`     | Apply rotation        | `cublasSrot` / `cublasDrot`     |

### Level 2 (Matrix-Vector)

| Method          | Description         | CUDA Equivalent               |
| --------------- | ------------------- | ----------------------------- |
| `sgemv / dgemv` | y = α·op(A)·x + β·y | `cublasSgemv` / `cublasDgemv` |
| `ssymv / dsymv` | Symmetric MV        | `cublasSsymv` / `cublasDsymv` |
| `strmv / dtrmv` | Triangular MV       | `cublasStrmv` / `cublasDtrmv` |
| `strsv / dtrsv` | Triangular solve    | `cublasStrsv` / `cublasDtrsv` |
| `ssyr / dsyr`   | Rank-1 update       | `cublasSsyr` / `cublasDsyr`   |

### Level 3 (Matrix-Matrix)

| Method                                      | Description                | CUDA Equivalent               |
| ------------------------------------------- | -------------------------- | ----------------------------- |
| `sgemm / dgemm`                             | C = α·A·B + β·C            | `cublasSgemm` / `cublasDgemm` |
| `sgemmStridedBatched / dgemmStridedBatched` | Strided batched GEMM       | `cublasSgemmStridedBatched`   |
| `sgemmBatched / dgemmBatched`               | Pointer-array batched GEMM | `cublasSgemmBatched`          |
| `gemmEx`                                    | Mixed-precision GEMM       | `cublasGemmEx`                |
| `gemmGroupedBatchedEx`                      | Grouped batched GEMM       | `cublasGemmGroupedBatchedEx`  |
| `ssymm / dsymm`                             | Symmetric MM               | `cublasSsymm` / `cublasDsymm` |
| `strsm / dtrsm`                             | Triangular solve           | `cublasStrsm` / `cublasDtrsm` |
| `ssyrk`                                     | Rank-k update              | `cublasSsyrk`                 |
| `strmm`                                     | Triangular MM              | `cublasStrmm`                 |
| `sgeam / dgeam`                             | Matrix add                 | `cublasSgeam` / `cublasDgeam` |
| `sdgmm / ddgmm`                             | Diagonal MM                | `cublasSdgmm` / `cublasDdgmm` |

### Enums

| Enum        | Values                                           |
| ----------- | ------------------------------------------------ |
| `Operation` | `.no_transpose`, `.transpose`, `.conj_transpose` |
| `FillMode`  | `.lower`, `.upper`                               |
| `SideMode`  | `.left`, `.right`                                |
| `DiagType`  | `.non_unit`, `.unit`                             |
| `DataType`  | `.f16`, `.bf16`, `.f32`, `.f64`                  |

---

## cuBLAS LT

**Import:** `const cublaslt = @import("zcuda").cublaslt;`
**Enable:** `-Dcublaslt=true`
📖 [Full documentation](cublaslt/README.md)

### CublasLtContext

| Method                                                                                       | Description               |
| -------------------------------------------------------------------------------------------- | ------------------------- |
| `init(ctx) !CublasLtContext`                                                                 | Create handle             |
| `deinit(self)`                                                                               | Destroy handle            |
| `createMatmulDesc(compute, scale) !desc`                                                     | Create matmul descriptor  |
| `createMatrixLayout(dtype, rows, cols, ld) !layout`                                          | Create matrix layout      |
| `setTransA(self, desc, op) !void`                                                            | Set transpose A           |
| `setTransB(self, desc, op) !void`                                                            | Set transpose B           |
| `getHeuristics(desc, la, lb, lc, ld, pref, results) !i32`                                    | Query algorithms          |
| `matmul(T, desc, α, A, la, B, lb, β, C, lc, D, ld, stream) !void`                            | Execute matmul            |
| `matmulWithAlgo(T, desc, α, A, la, B, lb, β, C, lc, D, ld, algo, ws, ws_size, stream) !void` | Matmul with explicit algo |

### Types

| Type          | Description                                      |
| ------------- | ------------------------------------------------ |
| `DataType`    | `.f16`, `.f32`, `.f64`, `.bf16`, `.i8`, `.i32`   |
| `ComputeType` | `.f32`, `.f64`, `.f16`, `.i32`, `.f32_fast_tf32` |
| `Operation`   | `.none`, `.transpose`, `.conjugate_transpose`    |

---

## cuDNN

**Import:** `const cudnn = @import("zcuda").cudnn;`
**Enable:** `-Dcudnn=true`
📖 [Full documentation](cudnn/README.md)

### CudnnContext

| Method                                                        | Description            |
| ------------------------------------------------------------- | ---------------------- |
| `init(ctx) !CudnnContext`                                     | Create cuDNN handle    |
| `deinit(self)`                                                | Destroy handle         |
| `createTensor4d(format, dtype, n, c, h, w) !TensorDescriptor` | Create 4D tensor       |
| `createTensorNd(dtype, dims, strides) !TensorDescriptorNd`    | Create N-D tensor      |
| `createFilter4d(dtype, format, k, c, h, w) !FilterDescriptor` | Create 4D filter       |
| `createFilterNd(dtype, format, dims) !FilterDescriptorNd`     | Create N-D filter      |
| `createConv2d(...) !ConvolutionDescriptor`                    | 2D convolution params  |
| `createConvNd(...) !ConvolutionDescriptorNd`                  | N-D convolution params |
| `convForward(T, α, x, w, conv, algo, ws, β, y) !void`         | Conv forward           |
| `convForwardNd(T, α, x, w, conv, algo, ws, β, y) !void`       | N-D conv forward       |
| `convBackwardData(...) !void`                                 | Conv backward (data)   |
| `convBackwardFilter(...) !void`                               | Conv backward (filter) |
| `activationForward(T, act, α, x, β, y) !void`                 | Activation forward     |
| `activationBackward(T, act, α, y, dy, x, β, dx) !void`        | Activation backward    |
| `softmaxForward(T, algo, mode, α, x, β, y) !void`             | Softmax forward        |
| `softmaxBackward(T, algo, mode, α, y, dy, β, dx) !void`       | Softmax backward       |
| `poolingForward(T, pool, α, x, β, y) !void`                   | Pooling forward        |
| `poolingBackward(T, pool, α, y, dy, x, β, dx) !void`          | Pooling backward       |
| `opTensor(T, op, α1, a, α2, b, β, c) !void`                   | Element-wise op        |
| `addTensor(T, α, a, β, c) !void`                              | C = α·A + β·C          |
| `scaleTensor(T, desc, y, α) !void`                            | Y = α·Y                |
| `reduceTensor(T, op, α, a, β, c, ws) !void`                   | Tensor reduction       |
| `getReductionWorkspaceSize(op, a_desc, c_desc) !usize`        | Reduction workspace    |
| `getConvForwardWorkspaceSize(...) !usize`                     | Conv workspace size    |

### Enums

`TensorFormat`, `DnnDataType`, `ConvMode`, `ConvFwdAlgo`, `ActivationMode`,
`PoolingMode`, `SoftmaxAlgo`, `SoftmaxMode`, `ReduceOp`, `OpTensorOp`

---

## cuFFT

**Import:** `const cufft = @import("zcuda").cufft;`
**Enable:** `-Dcufft=true`
📖 [Full documentation](cufft/README.md)

### CufftPlan

| Method                                  | Description           |
| --------------------------------------- | --------------------- |
| `plan1d(nx, type, batch) !CufftPlan`    | Create 1D FFT plan    |
| `plan2d(nx, ny, type) !CufftPlan`       | Create 2D FFT plan    |
| `plan3d(nx, ny, nz, type) !CufftPlan`   | Create 3D FFT plan    |
| `planMany(...) !CufftPlan`              | Advanced batched plan |
| `deinit(self)`                          | Destroy plan          |
| `getSize(self) !usize`                  | Query workspace size  |
| `setStream(self, stream) !void`         | Set CUDA stream       |
| `execC2C / execZ2Z(in, out, dir) !void` | Complex-to-complex    |
| `execR2C / execD2Z(in, out) !void`      | Real-to-complex       |
| `execC2R / execZ2D(in, out) !void`      | Complex-to-real       |

### Enums

| Enum        | Values                                                                 |
| ----------- | ---------------------------------------------------------------------- |
| `FftType`   | `.c2c_f32`, `.c2c_f64`, `.r2c_f32`, `.c2r_f32`, `.r2c_f64`, `.c2r_f64` |
| `Direction` | `.forward`, `.inverse`                                                 |

---

## cuRAND

**Import:** `const curand = @import("zcuda").curand;`
**Enable:** `-Dcurand=true`
📖 [Full documentation](curand/README.md)

### CurandContext

| Method                                          | Description                 |
| ----------------------------------------------- | --------------------------- |
| `init(ctx, rng_type) !CurandContext`            | Create generator            |
| `deinit(self)`                                  | Destroy generator           |
| `setSeed(self, seed) !void`                     | Set seed                    |
| `setOffset(self, offset) !void`                 | Set generator offset        |
| `setDimensions(self, n) !void`                  | Set quasi-random dimensions |
| `setStream(self, stream) !void`                 | Set CUDA stream             |
| `fillUniform(data) !void`                       | Uniform (0, 1] float        |
| `fillUniformDouble(data) !void`                 | Uniform (0, 1] double       |
| `fillNormal(data, mean, stddev) !void`          | Normal float                |
| `fillNormalDouble(data, mean, stddev) !void`    | Normal double               |
| `fillLogNormal(data, mean, stddev) !void`       | Log-normal float            |
| `fillLogNormalDouble(data, mean, stddev) !void` | Log-normal double           |
| `fillUniformU32(data) !void`                    | Uniform random u32          |
| `fillPoisson(data, lambda) !void`               | Poisson distribution        |

### Enums

| Enum      | Values                                                                                                        |
| --------- | ------------------------------------------------------------------------------------------------------------- |
| `RngType` | `.default`, `.xorwow`, `.mrg32k3a`, `.mtgp32`, `.mt19937`, `.philox4_32_10`, `.sobol32`, `.scrambled_sobol32` |

---

## cuSOLVER

**Import:** `const cusolver = @import("zcuda").cusolver;`
**Enable:** `-Dcusolver=true`
📖 [Full documentation](cusolver/README.md)

> **`info` parameter:** All cuSOLVER functions take `info: driver.CudaSlice(i32)` — a
> **device-side** pointer required by the cuSOLVER API. After calling `ctx.synchronize()`,
> copy the value to host with `stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info)`.

```zig
var d_info = try stream.allocZeros(i32, allocator, 1);
defer d_info.deinit();
var h_info: i32 = 0;

try sol.sgetrf(n, n, d_A, n, d_ws, d_ipiv, d_info); // device pointer
try ctx.synchronize();
try stream.memcpyDtoH(i32, @as(*[1]i32, &h_info), d_info); // bring to host
```

### CusolverDnContext

| Method                                  | Description               |
| --------------------------------------- | ------------------------- |
| `init(ctx) !CusolverDnContext`          | Create handle             |
| `deinit(self)`                          | Destroy handle            |
| `sgetrf_bufferSize / dgetrf_bufferSize` | LU workspace query        |
| `sgetrf(m,n,A,lda,ws,ipiv,info) !void` | LU factorization: PA = LU |
| `sgetrs(n,nrhs,A,lda,ipiv,B,ldb,info) !void` | LU solve: AX = B |
| `sgesvd_bufferSize / dgesvd_bufferSize` | SVD workspace query       |
| `sgesvd(jobu,jobvt,m,n,A,lda,S,U,ldu,VT,ldvt,ws,lwork,info) !void` | SVD: A = UΣVᵀ |

### CusolverDnExt

| Method                                    | Description              |
| ----------------------------------------- | ------------------------ |
| `init(base) CusolverDnExt`                | Wrap base context        |
| `spotrf / dpotrf`                         | Cholesky: A = LLᵀ        |
| `spotrs / dpotrs`                         | Cholesky solve           |
| `sgeqrf / dgeqrf`                         | QR factorization         |
| `sorgqr / dorgqr`                         | Extract Q matrix         |
| `ssyevd / dsyevd`                         | Eigenvalue decomposition |
| `sgesvdj / dgesvdj`                       | Jacobi SVD               |
| `sgesvdj_bufferSize / dgesvdj_bufferSize` | Jacobi SVD workspace     |

### GesvdjInfo

| Method                          | Description               |
| ------------------------------- | ------------------------- |
| `init() !GesvdjInfo`            | Create Jacobi SVD params  |
| `deinit(self)`                  | Destroy params            |
| `setTolerance(self, tol) !void` | Set convergence tolerance |
| `setMaxSweeps(self, n) !void`   | Set max iterations        |

### Enums

| Enum       | Values                  |
| ---------- | ----------------------- |
| `EigMode`  | `.no_vector`, `.vector` |
| `FillMode` | `.lower`, `.upper`      |

---

## cuSPARSE

**Import:** `const cusparse = @import("zcuda").cusparse;`
**Enable:** `-Dcusparse=true`
📖 [Full documentation](cusparse/README.md)

### CusparseContext

| Method                                                                     | Description              |
| -------------------------------------------------------------------------- | ------------------------ |
| `init(ctx) !CusparseContext`                                               | Create handle            |
| `deinit(self)`                                                             | Destroy handle           |
| `createCsr(rows, cols, nnz, row_offsets, col_indices, values) !SpMatDescr` | Create CSR matrix        |
| `createCoo(rows, cols, nnz, row_indices, col_indices, values) !SpMatDescr` | Create COO matrix        |
| `createCsrForSpGEMM(...) !SpMatDescr`                                      | Create CSR for SpGEMM    |
| `createDnVec(data) !DnVecDescr`                                            | Create dense vector      |
| `createDnMat(rows, cols, ld, values, dtype, order) !DnMatDescr`            | Create dense matrix      |
| `spMVBufferSize(op, α, A, x, β, y) !usize`                                 | SpMV workspace           |
| `spMV(op, α, A, x, β, y, workspace) !void`                                 | Sparse × dense vector    |
| `spMMBufferSize(opA, opB, α, A, B, β, C) !usize`                           | SpMM workspace           |
| `spMM(opA, opB, α, A, B, β, C, workspace) !void`                           | Sparse × dense matrix    |
| `createSpGEMMDescr(self) !SpGEMMDescriptor`                                | Create SpGEMM descriptor |
| `spGEMM_workEstimation(...) !void`                                         | SpGEMM work estimation   |
| `spGEMM_compute(...) !void`                                                | SpGEMM compute           |
| `spGEMM_copy(...) !void`                                                   | SpGEMM finalize          |

### Enums

| Enum              | Values                                                    |
| ----------------- | --------------------------------------------------------- |
| `Operation`       | `.non_transpose`, `.transpose`, `.conjugate_transpose`    |
| `SpGEMMAlgorithm` | `.default`, `.csr_deterministic`, `.csr_nondeterministic` |

---

## NVRTC

**Import:** `const nvrtc = @import("zcuda").nvrtc;`
📖 [Full documentation](nvrtc/README.md)

| Function                                              | Description                |
| ----------------------------------------------------- | -------------------------- |
| `compilePtx(allocator, src) ![]u8`                    | Compile CUDA C++ → PTX     |
| `compilePtxWithOptions(allocator, src, opts) ![]u8`   | Compile with options       |
| `compileCubin(allocator, src) ![]u8`                  | Compile CUDA C++ → CUBIN   |
| `compileCubinWithOptions(allocator, src, opts) ![]u8` | Compile CUBIN with options |
| `getVersion() !NvrtcVersion`                          | Get NVRTC version          |

---

## NVTX

**Import:** `const nvtx = @import("zcuda").nvtx;`
**Enable:** `-Dnvtx=true`
📖 [Full documentation](nvtx/README.md)

### Global Functions

| Function               | Description                  |
| ---------------------- | ---------------------------- |
| `rangePush(name) void` | Push named range marker      |
| `rangePop() void`      | Pop top range marker         |
| `mark(name) void`      | Place marker at current time |

### ScopedRange

| Method                   | Description           |
| ------------------------ | --------------------- |
| `init(name) ScopedRange` | Push range (RAII)     |
| `deinit(self) void`      | Pop range (via defer) |

### Domain

| Method                | Description         |
| --------------------- | ------------------- |
| `create(name) Domain` | Create named domain |
| `destroy(self) void`  | Destroy domain      |

---

## Kernel DSL

> **This is a separate module** — `@import("zcuda_kernel")`, not `@import("zcuda")`.
> Full reference: 📖 [docs/kernel/API.md](kernel/API.md)
> Migration guide: 📖 [docs/kernel/MIGRATION.md](kernel/MIGRATION.md)

The Kernel DSL lets you write CUDA GPU kernels in **pure Zig** — no CUDA C++, no `nvcc`.
Zig compiles them directly to PTX at `zig build` time via the LLVM NVPTX backend.

```zig
// A pure-Zig GPU kernel
export fn saxpy(n: u32, alpha: f32, x: [*]f32, y: [*]f32) callconv(.Kernel) void {
    var iter = cuda.types.gridStrideLoop(n);
    while (iter.next()) |i| {
        y[i] = alpha * x[i] + y[i];
    }
}
```

### Sub-modules (`src/kernel/`)

| File | Description |
| ---- | ----------- |
| `intrinsics.zig` | 98 inline fns: `threadIdx`, `blockIdx`, atomics (`atomicAdd`–`atomicDec`), warp shuffle/vote/match/reduce, fast math, bit ops, cache hints, type conversion, `__nanosleep`, `__byte_perm` |
| `tensor_core.zig` | 56 inline fns: WMMA (sm_70+), MMA PTX (sm_80+), FP8 (sm_89+), wgmma/TMA/cluster (sm_90+), tcgen05 (sm_100+) |
| `shared_mem.zig` | `SharedArray(T,N)` static SMEM (addrspace(3)), `dynamicShared(T)`, `dynamicSharedBytes()`, `clearShared`, `loadToShared`, `storeFromShared`, `reduceSum` |
| `arch.zig` | `SmVersion` enum (sm_52–sm_100+), `requireSM` comptime guard, `atLeast`, `codename` |
| `types.zig` | `DeviceSlice(T)` (get/set/len), `DevicePtr(T)` (load/store/atomicAdd), `GridStrideIterator`, `globalThreadIdx`, `gridStride` |
| `debug.zig` | `assertf`, `assertInBounds`, `safeGet`, `ErrorFlag` (5 error codes), `setError`, `checkNaN`, `printf`, `CycleTimer`, `__trap`, `__brkpt` |
| `device.zig` | Module root (re-exports all sub-modules as `cuda.*`) |
| `shared_types.zig` | Host-device shared: `Vec2/3/4`, `Int2/3`, `Matrix3x3/4x4`, `LaunchConfig` (init1D/2D, forElementCount) |
| `bridge_gen.zig` | `init(Config)` → comptime `Fn` enum, `load`, `loadFromPtx`, `getFunction`, `getFunctionByName` |

### Key API Summary

| Category | Examples |
| -------- | -------- |
| Thread indexing | `cuda.threadIdx()`, `cuda.blockIdx()`, `cuda.blockDim()`, `cuda.gridDim()`, `cuda.types.globalThreadIdx()` |
| Synchronization | `cuda.__syncthreads()`, `cuda.__syncwarp(mask)`, `cuda.__threadfence()`, `cuda.__syncthreads_count/and/or` |
| Atomics | `atomicAdd`, `atomicSub`, `atomicMin`, `atomicMax`, `atomicCAS`, `atomicExch`, `atomicAnd/Or/Xor`, `atomicInc/Dec`, `atomicAdd_f64` |
| Warp shuffles | `__shfl_sync`, `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync` |
| Warp vote/match | `__ballot_sync`, `__all_sync`, `__any_sync`, `__activemask`, `__match_any_sync`, `__match_all_sync` |
| Warp reduce (sm_80+) | `__reduce_add_sync`, `__reduce_min/max/and/or/xor_sync` |
| Fast math | `__sinf`, `__cosf`, `__tanf`, `rsqrtf`, `sqrtf`, `__expf`, `__logf`, `__fmaf_rn`, `__fdividef`, `__saturatef`, `__powf` |
| Integer ops | `__clz`, `__clzll`, `__popc`, `__popcll`, `__brev`, `__brevll`, `__ffs`, `__byte_perm`, `__dp4a`, `__dp2a_lo/hi` |
| Cache hints | `__ldg`, `__ldca`, `__ldcs`, `__ldcg`, `__stcg`, `__stcs`, `__stwb` |
| Type conversion | `__float2int_rn/rz`, `__float_as_int`, `__int_as_float`, `__double2hiint`, `__hiloint2double` |
| Shared memory | `SharedArray(T,N)`, `dynamicShared(T)`, `clearShared`, `loadToShared`, `reduceSum` |
| Device types | `DeviceSlice(T).get/set/len`, `DevicePtr(T).atomicAdd`, `GridStrideIterator`, `gridStrideLoop(n)` |
| Shared types | `Vec2/3/4`, `Int2/3`, `Matrix3x3/4x4`, `LaunchConfig.forElementCount` |
| Debug | `assertf`, `ErrorFlag`, `printf`, `CycleTimer`, `__trap` |
| Clock | `cuda.clock()`, `cuda.clock64()`, `cuda.globaltimer()`, `cuda.__nanosleep(ns)` |
| WMMA | `wmma_load_a_f16`, `wmma_mma_f16_f32`, `wmma_store_d_f32`, `wmma_mma_s8_s32` |
| MMA PTX | `mma_f16_f32`, `mma_bf16_f32`, `mma_tf32_f32`, `mma_f64_f64`, `mma_e4m3_f32` |
| cp.async / wgmma / TMA | `memcpy_async`, `cp_async_wait_all`, `wgmma_f16_f32`, `tma_load`, `bulk_copy_g2s` |
| Bridge | `init(Config)`, `load(ctx, allocator)`, `getFunction(mod, .funcName)` |
