---
layout: post
title: "[ML] -CUDA-Introduction"
date: 2026-01-1113:19
subtitle:
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
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
