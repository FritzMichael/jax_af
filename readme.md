# Repository for Implementations of the Differentiable Backprojection Kernels for SAR applications

This repository contains implementations of the differentiable backprojection kernels for SAR applications using JAX and Warp.
The implementations highlight the advantages of using a vector jacobian product (JVP) with explicit rematerialization of the forward pass.
Explicit recalculation is faster than storing intermediate results because both passes become memory bound if they have to store/load intermediate results from global memory.
Additionally, the intermediate results scale with the number of chirps and the grid size, which quickly saturates the memory of even data center GPUs.
Instead the intermediate results can be recomputed on the fly from the input data, which just scales with the slow time extent of the data, which fits into L2 cache of the GPU and stays hot in-betweeen gradient-descent iterations.

# Outlook

While Warp with an explicit backward pass is quite fast, one can further gain performance by dropping to a CUDA C++ implementation or selectively using PTX intrinsics in the Warp implementation for features like __sincosf, and __ldg, or asynchronous pre-fetching of the input data into shared memory.

# Benchmarks

Benchmarks conducted on an Nvidia RTX 5070 Ti, 1024 chirps on 512² grids.
Reported numbers are median values over 100 iterations profiled with Nsight Systems.

| Method               | fwd (ms) | bwd (ms) | Peak Memory |
|----------------------|----------|----------|-------------|
| `jax.grad`           | 32.749   | 34.239   | ⚠️ 8.07 GB  |
| `jax.vjp`            | 10.602   | 30.232   | 75 MB       |
| `warp (AD)`          | 2.797    | 93.194   | 150 MB      |
| `warp (explicit bwd)`| 2.717    | 3.461    | 82 MB       |

# Benchmark artifacts

Nvidia Nsight Systems and Nsight Compute were used to profile and guide the optimizations of the implementations.
Nsight Systems reports are available in the benchmark_artifacts folder.