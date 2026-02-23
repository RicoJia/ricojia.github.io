---
layout: post
title: "[CUDA - 2] CUDA Introduction"
date: 2026-01-14 13:19
subtitle: First CUDA Program, SIMD SIMT,
comments: true
tags:
  - CUDA
---

## What is Cuda

Cuda is fundamentally C++ language with extensiions. kernels, `__global__`, `__device__` etc are defined in the C++ space. If you want to expose C to CUDA, just follow the standard

```c++
extern "C" void launch_kernel(...);
```

So this means

- You can use host-side containers like `std::vector`,  then pass its `.data()` pointer to `cudaMalloc`/`cudaMemcpy` or to kernels. But you cannot use `std::vector` directly on the GPU device from device code.
- For device-side containers use libraries designed for CUDA, e.g. `thrust::device_vector`, `cub`, or manage raw device pointers yourself.

`nvcc` is a compiler driver CUDA uses and is not a single compiler. It splits `.cu` into:

- host code (compiled by your host compiler, like `g++`)
- device code (compiled by NVIDIA's device toolchain, `PTX + SASS`)

`ATen` is a C++ tensor library `PyTorch` and `libtorch` uses to manipulate tensors. Autograd is on top of `ATen`.

- It provides `at::Tensor`, core tensor operations, device CPU/CUDA handling, and backend dispatch mechanism
- `#include <ATen/ATen.h>`
- In PyTorch/ATen extensions **you usually work with `at::Tensor` on the host** and **pass raw pointers (`tensor.data_ptr<T>()`) into CUDA kernels**; ATen handles *CPU/CUDA dispatch and memory details*.

## How to Compile

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

### SIMT and SIMD

CUDA is a very good embodiment of SIMD (Single-Instruction-Multiple Data). SIMD is great for addressing **embarassingly parallel problems**, problems that are so "embarassingly" simple, that there are no dependency on each other.

- One example is point cloud transformation. All points can be transformed into a different frame by MMA with transform matrices.

Up until 2016, a CUDA core does SIMD. In SIMD, the same exact instructions are applied to the same data, like a marching band

```
C[i] = A[i] * B[i]
```

However, when there is a condition in the kernel, there is a divergence. In the below example, the Stream Manager would split this logic into two passes for threads, one for `C[i] = A[i] * B[i]`, and one for `[i] = A[i] + B[i]`

```
if A[i] > 0:
    C[i] = A[i] * B[i]
else:
    C[i] = A[i] + B[i]
```

Single-Instruction-Multiple-Threads SIMT can handle moderate levels of logical branches **on the thread level**. That is, it would take 1 pass for an arbitrary thread to decide which instruction it should execute. This means each thread has a **program counter**. Meanwhile, there are 128kb L1 caches for 1 thread to share data with others.

```c
__global__ void conditional_op(float* A, float* B, float* C, int n) {
    // Compute the unique thread ID
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure we don’t go out of bounds
    if (idx < n) {
        if (A[idx] > 0) {
            C[idx] = A[idx] * B[idx];  // Branch 1
        } else {
            C[idx] = A[idx] + B[idx];  // Branch 2
        }
    }
}
```

For SIMT:

- The more branches, the less performance. Nested logic in a kernel is not a good idea
- Contiguous Memory (coalesced) are faster to access in a **warp** of memory
