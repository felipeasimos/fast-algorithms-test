# GEMM

General Matrix Multiplication.

The functions generally take the matrices `C`, `A` and `B`, alongside `ni`, `nj` and `nk` as arguments. The arguments relate to each other in this way:

```
C = A B
ni = number of rows in A and C
nj = number of columns in B and C
nk = number of columns in A and number of rows in B

C is (ni, nj)
A is (ni, nk)
B is (nk, nj)
```

### GEMM Blocked with AVX (single-thread)

The majors matter a lot when trying to vectorize since we need contigous memory operate on. The optimal solution is to avoid having to reduce vectors to a single scalars (reduction to get the sum), by partially computing elements of C.

A subtle point is that A and B share the `nk` dimension, so that is what we iterate over at the innermost loop. This means that optimally `A` iterate over columns and `B` over rows.

1. With blocked gemm, with blocksize at the size of a cache line, we have total control of the memory major of our blocks while optimizing cache access during packing.
2. `C` is not usually packed, so we have to choose the memory layout around it:
   * RRR: broadcast A, vectorize a B row and partially compute a row of C
   * CCR: vectorize an A column, broadcast B and partially compute a column of C
      * faster, iterate over `k` in the innermost loop for B and A

### CUDA vs WebGPU


| Concept         | CUDA                                                     | WGPU (WebGPU / wgpu-native)                                 |
| --------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| Execution model | Grid of thread blocks, each with many threads            | Workgroups of invocations (threads)                         |
| Language        | CUDA C (extension of C++)                                | WGSL (WebGPU Shading Language, similar to GLSL)             |
| Launch          | `kernel<<<gridDim, blockDim>>>(args)`                    | `dispatch_workgroups(x, y, z)`                              |
| Thread index    | `threadIdx.x + blockIdx.x * blockDim.x`                  | `@builtin(global_invocation_id)`                            |
| Memory types    | `__global__`, `__shared__`, `__device__`, `__constant__` | Storage buffers, workgroup memory, uniforms                 |
| Synchronization | `__syncthreads()`                                        | `workgroupBarrier()`                                        |
| Host side       | CUDA runtime API (cudaMalloc, cudaMemcpy, etc.)          | wgpu API for creating buffers, pipelines, binding, dispatch |
| Kernel          | `__global__ void myKernel(...)`                          | `@compute @workgroup_size(...) fn main(...)`                |
