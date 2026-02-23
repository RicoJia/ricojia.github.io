---
layout: post
title: "[CUDA - 2] Introduction to CUDA Coding"
date: 2026-01-14 13:19
subtitle: First CUDA Program, JIT Compile
comments: true
tags:
  - CUDA
---
## What is Cuda

CUDA is fundamentally C++ language with extensions. kernels, `__global__`, `__device__` etc are defined in the C++ space. If you want to expose C to CUDA, just follow the standard

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

---

### Programming Hierarchy

The hierarchy:

```c
Grid → Blocks → Threads
```

- grid: a collection of many blocks. **blocks are independent of each other**.
- block: a group of threads (max 1024) that can coorperate (shared memory + sync)
 	- block is a **software-visible** grouping
 	- A warp is a group of 32 threads, which are schedulred and executed together by the hardware. A warp is the **smallest unit** the GPU actually runs.
 	- So if you launch 512 threads in a block, it actually splits into `512 / 32 = 16 warps`  .
 	- So, `threadIdx.x = 0–31 → warp 0`, `threadIdx.x = 32–63 → warp 1`
- thread: executes kernel code independently
 	- threads inside the same block **can share memory**
 	- can synchronize via `__syncthreads()`

thread indexing:

```
thread_id = threadIdx.x + blockDim.x ∗ threadIdx.y + blockDim.x ∗ blockDim.y ∗ threadIdx.z
```

- `__shared__` : shared memory (per block) on GPU

### Why Warps Matter

Warp executes in SIMT style. All 32 threads execute the same instruction at the same time. If you write

```c++
if (threadIdx.x < 16) do_A();
else do_B();
```

The **warp must execute both path serially**. This is called **Warp-Divergence** (which could slow down GPU performance).

### Memory Access Efficiency

If 32 threads in a warp access:

```cpp
xyz[i*3 + 0]  
xyz[i*3 + 1]  
xyz[i*3 + 2]
```

**in contiguous memory, the GPU can coalesce it into one big memory transaction**. If accesses are scattered, memory access **will be slower**.

CUDA threads have built-in IDs:

- `blockIdx`, `threadIdx`, `blockDim`, `gridDim`

### Kernel and Variables

A kernel is just a function that runs on the GPU. You don't call it like a normal function, but instead you launch it over a grid of many lightweight threads. Each thread computes a small piece of the overall work.

A kernel must have a **function qualifier** which indicates where the kernel is run:

```cpp
__global__ void NmDistanceKernel(...) { ... }
__device__ float sq(float a) { return a*a; }
__host__ void foo() { ... }  // same as just: void foo() { ... }

__host__ __device__ inline float clamp(float x, float lo, float hi) {  
  return x < lo ? lo : (x > hi ? hi : x);  
}
```

- `__global__` means the kernel is called from the CPU, runs on the GPU. Return Type must be `void`. This is what **makes it a kernel**: it could launched with `<<<grid, block>>>`
- `__device__` : called from the GPU, runs on the GPU. It's not launchable with `<<< >>>` from the CPU, but instead it runs on other kernels?
- `__host__`: called from the CPU, runs on the CPU. It's basically a normal C++ function
- combinations: `__host__ __device__`: this is compiled into two versions of the function, one for CPU calls, one for GPU calls. It's useful for small utilities in both places.
 	- In such a function, you can't freely use host-only code like `printf`, `new`, os calls, **unless guarded**

Variable specifiers:

```cpp
__constant__ float LUT[256];  
__shared__ float tile[256];
```

- `__shared__` : shared memory (per block) on GPU, slower
- `__device__`: global device variable (lives on GPU), faster
- `__constant__`: constant memory on GPU (cached, read-only from kernels)
- `__managed__` : unified memory (accessible from CPU & GPU; managed migration)

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
