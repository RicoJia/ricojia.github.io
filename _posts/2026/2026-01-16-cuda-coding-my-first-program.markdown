---
layout: post
title: "[CUDA - 2] My First CUDA Kernel - Chamfer Distance"
date: 2026-01-16 13:19
subtitle: First CUDA Program, JIT Compile
comments: true
tags:
  - CUDA
---
## My First CUDA Program

A kernel launch looks like

```cpp
Kernel<<<gridDim, blockDim>>>(...)
```

Inside kernels, you see built-in indices:

```cpp
blockIdx.x, blockIdx.y, blockIdx.z
threadIdx.x, threadIdx.y, threadIdx.z
gridDim.*, blockDim.*: sizes
```

Warp and Streaming Processors:

- threads are executed in groups of 32 threads, each group is a **warp**. Warps are schedulred onto SMs (streaming processors). Threads in the same warp execute the same instruction (SIMT). If they branch differently, performance can drop.

Memory types:

- global memory: large, slowish, accessible by all threads. Your tensor data lives here.
- shared memory: small, fast, per-block, used for coorperation (your `__shared__ float buf[...]`)
- registers: fastest, per thread (your local float x1, y1, z1)
- `__syncthreads()`: block level barrier, all threads in the block wait until everyone reaches it.

```cpp
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <stdio.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

/**
 * point cloud 1: (B, N, 3), point cloud 2: (B, M, 3)
 * one block handles one batch element: i = blockIdx.x
 * threads in the block handle source points: j = threadIdx.x
 */
__global__ void nearest_neighbor_kernel(int batch_size, int src_point_cloud_size, const float* __restrict__ src_points, int dst_point_cloud_size,
                                        const float* __restrict__ dst_points, float* __restrict__ result, int* __restrict__ result_idx)
{
    // 1. Copy dst_points into shared memory in tiles, from all threads
    // This is because reading points2 from global memory is slow
    // shared memory is much faster
    // const / constexpr defines a per-thread local variable, constexpr usually gets compiled into an expression through substitution;
    // const will be held in a register at worst
    // A tile is a contiguous subgroup of the dst point cloud
    constexpr int tile_size = 512;  // number of points per tile. 512 *3 floats = 6KB < 48KB shared memory limit

    // Does every thread launch this kernel??
    // Is this buffer created per block???
    __shared__ float dst_points_buf[tile_size * 3];

    const int batch_idx = blockIdx.x;
    const int pt_idx = blockIdx.y * blockDim.x + threadIdx.x;

    // Out-of-bounds threads must still participate in __syncthreads() to avoid
    // deadlock when n is not a multiple of blockDim.x. Use an active flag to
    // guard work instead of returning early.
    const bool active = (batch_idx < batch_size && pt_idx < src_point_cloud_size);

    float x1 = 0.f, y1 = 0.f, z1 = 0.f;
    float best_dist = 1e30f;
    int best_j = 0;

    if (active)
    {
        const int src_pt_idx = (batch_idx * src_point_cloud_size + pt_idx) * 3;
        x1 = src_points[src_pt_idx];
        y1 = src_points[src_pt_idx + 1];
        z1 = src_points[src_pt_idx + 2];
    }

    for (int tile_idx = 0; tile_idx < dst_point_cloud_size; tile_idx += tile_size)
    {
        const int num_pt_clouds_in_tile = min(tile_size, dst_point_cloud_size - tile_idx);

        // cooperative load: ALL threads participate to avoid __syncthreads() deadlock
        for (int t = threadIdx.x; t < num_pt_clouds_in_tile * 3; t += blockDim.x)
        {
            dst_points_buf[t] = dst_points[(batch_idx * dst_point_cloud_size + tile_idx) * 3 + t];
        }

        __syncthreads();

        // Only active threads search within the shared-memory tile
        if (active)
        {
            for (int k = 0; k < num_pt_clouds_in_tile; ++k)
            {
                const float x2 = dst_points_buf[k * 3 + 0];
                const float y2 = dst_points_buf[k * 3 + 1];
                const float z2 = dst_points_buf[k * 3 + 2];

                const float dx = x1 - x2;
                const float dy = y1 - y2;
                const float dz = z1 - z2;
                const float d = dx * dx + dy * dy + dz * dz;

                if (d < best_dist)
                {
                    best_dist = d;
                    best_j = tile_idx + k;  // index in dst cloud
                }
            }
        }

        __syncthreads();
    }

    if (active)
    {
        const int out_idx = batch_idx * src_point_cloud_size + pt_idx;
        result[out_idx] = best_dist;
        result_idx[out_idx] = best_j;
    }
}

/**
 * @brief: launcher function
 */
int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2)
{
    constexpr int THREADS_NUM = 512;
    const auto batch_size = xyz1.size(0);
    const auto n = xyz1.size(1);
    const auto m = xyz2.size(1);

    dim3 grid_size_src_1(batch_size, (n + THREADS_NUM - 1) / THREADS_NUM, 1);
    dim3 grid_size_src_2(batch_size, (m + THREADS_NUM - 1) / THREADS_NUM, 1);

    // one grid, with x (32) x y (16) blocks, each block with 512 threads
    // i.e., blockIdx.x ∈ [0,31] blockIdx.y ∈ [0,15]
    // Here grid.x is batch size, grid.y is the number of blocks needed in which there's THREADS_NUM threads.

    // 1. find best neighbor from points1 to points2
    // dist1[i,j] = min_k || A[i,j] - B[i,k] ||^2 what is i, j?
    // idx1[i,j] = argmin_k || A[i,j] - B[i,k] ||^2

    nearest_neighbor_kernel<<<grid_size_src_1, THREADS_NUM>>>(batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(), dist1.data_ptr<float>(),
                                                              idx1.data_ptr<int>());

    // 2. find best neighbor from points2 to points1
    // dist2[i,k] = min_j || B[i,k] - A[i,j] ||^2
    // idx2[i,k] = argmin_j || B[i,k] - A[i
    nearest_neighbor_kernel<<<grid_size_src_2, THREADS_NUM>>>(batch_size, m, xyz2.data_ptr<float>(), n, xyz1.data_ptr<float>(), dist2.data_ptr<float>(),
                                                              idx2.data_ptr<int>());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "error in nearest neighbor updateOutput: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return 1;
}

// dist1(i,j) = (x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2
// ∂dist1/∂x1 = 2*(x1 - x2)
// ∂dist1/∂x2 = -2*(x1 - x2) ...
// Then apply chain rule:
// ∂loss / ∂x += ∂loss/∂dist1 * ∂dist1/ ∂x1
__global__ void nearest_neighbor_grad_kernel(int batch_size, int src_point_cloud_size, const float* __restrict__ src_points, int dst_point_cloud_size,
                                             const float* __restrict__ dst_points, float* __restrict__ grad_dist1, const int* __restrict__ idx1,
                                             float* __restrict__ grad_src_points, float* __restrict__ grad_dst_points)
{
    // 1. each block is in charge of one batch
    const int batch_idx = blockIdx.x;
    const int pt_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || pt_idx >= src_point_cloud_size)
        return;

    int src_ptx_idx = (batch_idx * src_point_cloud_size + pt_idx) * 3;
    float x1 = src_points[src_ptx_idx];
    float y1 = src_points[src_ptx_idx + 1];
    float z1 = src_points[src_ptx_idx + 2];

    int j = idx1[batch_idx * src_point_cloud_size + pt_idx];  // index of the point in src point cloud
    if (j < 0 || j >= dst_point_cloud_size)                   // shouldn't happen, but just in case
        return;

    // dst point
    int dst_ptx_idx = (batch_idx * dst_point_cloud_size + j) * 3;
    float x2 = dst_points[dst_ptx_idx];
    float y2 = dst_points[dst_ptx_idx + 1];
    float z2 = dst_points[dst_ptx_idx + 2];

    float g = grad_dist1[batch_idx * src_point_cloud_size + pt_idx] * 2.0f;
    float gx1 = g * (x1 - x2);
    float gy1 = g * (y1 - y2);
    float gz1 = g * (z1 - z2);

    // writing to gradient of xyz1, which is guaranteed to be unique per thread. No atomic add needed
    grad_src_points[src_ptx_idx] += gx1;
    grad_src_points[src_ptx_idx + 1] += gy1;
    grad_src_points[src_ptx_idx + 2] += gz1;

    // writing to gradient of xyz2, which may have multiple threads writing to the same location. Atomic add needed
    atomicAdd(&grad_dst_points[dst_ptx_idx], -gx1);
    atomicAdd(&grad_dst_points[dst_ptx_idx + 1], -gy1);
    atomicAdd(&grad_dst_points[dst_ptx_idx + 2], -gz1);
}

int chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor grad_xyz1, at::Tensor grad_xyz2, at::Tensor graddist1, at::Tensor graddist2, at::Tensor idx1,
                          at::Tensor idx2)
{
    if (!xyz1.is_cuda() || !xyz2.is_cuda())
        return 0;
    if (!xyz1.is_contiguous() || !xyz2.is_contiguous())
        return 0;
    if (!grad_xyz1.is_contiguous() || !grad_xyz2.is_contiguous())
        return 0;
    if (!graddist1.is_contiguous() || !graddist2.is_contiguous())
        return 0;
    if (!idx1.is_contiguous() || !idx2.is_contiguous())
        return 0;

    // PyTorch may have tensors on different CUDA devices (GPU 0, GPU 1), here using RAII,
    // after the object goes out of scope, the previous device is restored
    c10::cuda::CUDAGuard device_guard(xyz1.device());

    const int batch_size = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    cudaMemsetAsync(grad_xyz1.data_ptr<float>(), 0, (size_t)batch_size * n * 3 * sizeof(float), stream);
    cudaMemsetAsync(grad_xyz2.data_ptr<float>(), 0, (size_t)batch_size * m * 3 * sizeof(float), stream);

    constexpr int THREADS_NUM = 512;

    dim3 grid_src1(batch_size, (n + THREADS_NUM - 1) / THREADS_NUM, 1);
    dim3 grid_src2(batch_size, (m + THREADS_NUM - 1) / THREADS_NUM, 1);

    // Launch 1: xyz1 -> xyz2 contributes to gradxyz1 and gradxyz2

    // <<<grid_src1, THREADS_NUM, 0, stream>>> 0 means no additional dynamic shared memory at run time
    // stream means the kernel gets launched in this stream, respecting the order of operations in this stream
    // If you omit it, you get the default stream, which may not respect in multi-stream scenarios
    nearest_neighbor_grad_kernel<<<grid_src1, THREADS_NUM, 0, stream>>>(batch_size, n, xyz1.data_ptr<float>(), m, xyz2.data_ptr<float>(), graddist1.data_ptr<float>(),
                                                                        idx1.data_ptr<int>(), grad_xyz1.data_ptr<float>(), grad_xyz2.data_ptr<float>());
    nearest_neighbor_grad_kernel<<<grid_src2, THREADS_NUM, 0, stream>>>(batch_size, m, xyz2.data_ptr<float>(), n, xyz1.data_ptr<float>(), graddist2.data_ptr<float>(),
                                                                        idx2.data_ptr<int>(), grad_xyz2.data_ptr<float>(), grad_xyz1.data_ptr<float>());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "error in nearest neighbor update gradient : " << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return 1;
}
```

### Export

In `chamfer_cuda.cpp`, we use pybind11 to bind two functions for the forward pass and backward pass:

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {  
m.def("forward", &chamfer_forward, "chamfer forward (CUDA)");  
m.def("backward", &chamfer_backward, "chamfer backward (CUDA)");  
}
```

- forward pass: compute nearest neighbor squared distances + argmin indices
- backward pass: compute gradients w.r.t point coordinates using those argmin indices

---

## How to Compile

### Compilation Method 1 - the classical setup.py

`setup.py`

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension('chamfer_3D', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer3D.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```

- `setuptools` register an extension with sources `chamfer_cuda.cpp` and `chamfer3D.cu`
- `BuildExtension` invokes host side C++ compiler (for `.cpp`) and nvcc (for `.cu` files) into object files, then link them together into a shared object `.so`
  - the object files are linked against PyTorch/ATen/c10 and CUDA runtime libraries (e.g., `libcudart`), so the `.so` is an importable Python extension
    - it's `-fPIC`, position independent code, so it can be linked into a shared library
    - Pybind11 or torch macros are used to expose functions to python
    - `kernels` in C++ are `at::Tensor::data_ptr<T>()` or `tensor.contiguous().data_ptr()`

 Then in a new console:

```bash
python3 setup.py build_ext --inplace
```

### Compilation Method 2: Pytorch JIT Compilation

If a `.so` is not found, one can use Pytorch to compile using `torch.utils.cpp_extension.load`, no setup.py or `build_ext` is needed.  You do need Ninja

```bash
apt-get install -y ninja-build
```

```python
from torch.utils.cpp_extension import load

chamfer_3D = load(
    name="chamfer_3D",
    sources=["chamfer_cuda.cpp", "chamfer3D.cu"]
)
```

What it does:

1. Calls `nvcc` + the host C++ compiler **the first time** the module is imported
2. Caches the compiled `.so` in `~/.cache/torch_extensions/` (keyed by source hash + PyTorch version)
3. Returns the extension as a **regular Python module** — `chamfer_3D.forward(...)` just works
4. On subsequent imports: cache hit → **instant load**, no recompile

---

## Gotchas

- **DEADLOCK-ALERT** If the code has some threads return before reaching  `__syncthreads()`, then there could be a deadlock

```cpp
if (threadIdx.x >= some_threshold) return;  
...  
__syncthreads();
```

This is because `__syncthreads()` is a block-wide barrier. **For the block to proceed, all threads in the block must arrive at the barrier**. This is one of the most common CUDA bugs.
