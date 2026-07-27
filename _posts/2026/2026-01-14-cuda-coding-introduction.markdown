---
layout: post
title: "[CUDA - 2] Introduction to CUDA Coding"
date: 2026-01-14 13:19
subtitle: CUDA Programming Hierarchy, Jetpack, MPS
comments: true
header-img: img/post-bg-alibaba.jpg
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

### Gotcha - cuda version is in /usr/local/cuda

  `nvidia-smi` shows the installed NVIDIA driver’s maximum supported CUDA runtime API version: 13.0. The wrapper build uses the local CUDA toolkit at /usr/local/cuda—compiler, headers, and CMake files—which is 12.2.

```
readlink -f /usr/local/cuda # see /usr/local/cuda-12.2
nvcc --version
```

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

Grid and Block Dimension limits

- gridDim.y <= 65535

```python
import torch

p = torch.cuda.get_device_properties(0)
print(f"Device:                      {p.name}")
print(f"Architecture:                {p.major}.{p.minor}")
print(f"SMs:                         {p.multi_processor_count}")
print(f"Warp size:                   {p.warp_size}")
print(f"Max threads per SM:          {p.max_threads_per_multi_processor}")
print(f"Regs per SM:                 {p.regs_per_multiprocessor}")
print(f"L2 cache:                    {p.L2_cache_size / 1024:.0f} KB")
print(f"Total global memory:         {p.total_memory / 1024**3:.1f} GB")

import ctypes

libcudart = ctypes.CDLL("libcudart.so")


class _CudaDeviceProp(ctypes.Structure):
    _fields_ = [
        ("name",                ctypes.c_char * 256),  # 0
        ("uuid",                ctypes.c_byte * 16),   # 256
        ("luid",                ctypes.c_byte * 8),    # 272
        ("luidDeviceNodeMask",  ctypes.c_uint),        # 280
        ("_pad",                ctypes.c_byte * 4),    # 284 → align next size_t to 288
        ("totalGlobalMem",      ctypes.c_size_t),      # 288
        ("sharedMemPerBlock",   ctypes.c_size_t),      # 296
        ("regsPerBlock",        ctypes.c_int),         # 304
        ("warpSize",            ctypes.c_int),         # 308
        ("memPitch",            ctypes.c_size_t),      # 312
        ("maxThreadsPerBlock",  ctypes.c_int),         # 320
        ("maxThreadsDim",       ctypes.c_int * 3),    # 324
        ("maxGridSize",         ctypes.c_int * 3),    # 336
        # rest of the struct — pad to 4096 so cudaGetDeviceProperties never overruns
        ("_rest",               ctypes.c_byte * 3748), # 348 → 4096
    ]

prop = _CudaDeviceProp()
libcudart.cudaGetDeviceProperties(ctypes.byref(prop), ctypes.c_int(0))

print(f"\n--- from CUDA runtime ---")
print(f"Max threads per block:       {prop.maxThreadsPerBlock}")
print(f"Max block dim:               ({prop.maxThreadsDim[0]}, {prop.maxThreadsDim[1]}, {prop.maxThreadsDim[2]})")
print(f"Max grid dim:                ({prop.maxGridSize[0]}, {prop.maxGridSize[1]}, {prop.maxGridSize[2]})")
print(f"Shared mem per block:        {prop.sharedMemPerBlock} bytes  ({prop.sharedMemPerBlock / 1024:.0f} KB)")
print(f"Regs per block:              {prop.regsPerBlock}")
```

My laptop gives:

```
Device:                      NVIDIA RTX A4000 Laptop GPU
Compute capability:          8.6
SMs:                         40
Warp size:                   32
Max threads per SM:          1536
Regs per SM:                 65536
L2 cache:                    4096 KB
Total global memory:         7.7 GB

--- from CUDA runtime ---
Max threads per block:       1024
Max block dim:               (1024, 1024, 64)
Max grid dim:                (2147483647, 65535, 65535)
Shared mem per block:        49152 bytes  (48 KB)
Regs per block:              65536
```

- grid size are in number of blocks.
- `Max block dim: 1024, 1024, 64)` just gives the maximum **per dimension**. The total number of threads per block can never exeed `Max threads per block: 1024`. E.g., `blockDim = (1024, 2, 1)   → 2048 threads`  `❌` exceeds `maxThreadsPerBlock`

### Why Warps Matter

Warp executes in SIMT style. All 32 threads execute the same instruction at the same time. If you write

```c++
if (threadIdx.x < 16) do_A();
else do_B();
```

The **warp must execute both path serially**. This is called **Warp-Divergence** (which could slow down GPU performance).

### Memory Access & Efficiency

If 32 threads in a warp access:

```cpp
xyz[i*3 + 0]  
xyz[i*3 + 1]  
xyz[i*3 + 2]
```

**in contiguous memory, the GPU can coalesce it into one big memory transaction**. If accesses are scattered, memory access **will be slower**.

CUDA threads have built-in IDs:

- `blockIdx`, `threadIdx`, `blockDim`, `gridDim`

To launch:

```cpp
kernel<<< dim3(grid_x, grid_y, grid_z),  
          dim3(block_x, block_y, block_z) >>>(...);
```

- `dim3(grid_x, grid_y, grid_z)` = **grid dimensions** = how many **blocks**
- `dim3(block_x, block_y, block_z)` = **block dimensions** = how many **threads per block**

Thread indexing:

```
thread_id = threadIdx.x + blockDim.x ∗ threadIdx.y + blockDim.x ∗ blockDim.y ∗ threadIdx.z
```

#### Shared Memory Limit

- **48 KB per block** on many GPUs by default
- Often configurable up to **96 KB** or **100 KB+** on newer architectures (if the GPU supports it and you opt-in)

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
__device__ __forceinline__ void reduce_max_pair(float* dists, int* dists_i, int i, int j) {}
```

- `__global__` means the kernel is called from the CPU, runs on the GPU. Return Type must be `void`. This is what **defines a kernel**: it must be launched with `<<<grid, block>>>`
  - The kernel must be `__global__ void`
  - The kernel is called once on the CPU with `<<<grid, block>>>`, but called on every thread.  
- `__device__` : called from the GPU, runs on the GPU. It's not launchable with `<<< >>>` from the CPU, but instead it runs on other kernels?
  - It's not a kernel, but **instead a GPU function**
- `__host__`: called from the CPU, runs on the CPU. It's basically a normal C++ function
- combinations: `__host__ __device__`: this is compiled into two versions of the function, one for CPU calls, one for GPU calls. It's useful for small utilities in both places.
  - In such a function, you can't freely use host-only code like `printf`, `new`, os calls, **unless guarded**
  - `__forceinline__`: inline the function aggressively instead of generating a function call.

Variable specifiers:

```cpp
__constant__ float LUT[256];  
__shared__ float tile[256];

__global__ void nearest_neighbor_kernel( const float* __restrict__ src_points)
```

- `__shared__` : shared memory (per block) on GPU, slower
  - when launched in a kernel, CUDA runtime will make sure it's created only ONCE per block.
- `__device__`: global device variable (lives on GPU), faster
- `__constant__`: constant memory on GPU (cached, read-only from kernels)
- `__managed__` : unified memory (accessible from CPU & GPU; managed migration)
- `__restrict__` is a **promise to the compiler** about pointers: "for the lifetime of the pointer, it does not overlap with the memory of any other `__restrict__` pointers"
  - Without it, the compiler would become conservative and add extra loads / stores, fewer reorderings.

---

## Jetpack

Jetpack: whole NVIDIA software stack for **Jetson**, not just GPU

```
JetPack = Ubuntu-based Jetson Linux
        + kernel / drivers / BSP
        + CUDA
        + cuDNN
        + TensorRT
        + VPI / DLA / multimedia / camera stack
        + Nsight tools
        + containers / dev tools
```

Jetpack is not on the GPU only, it affects:

- CPU: ubuntu, kernel, drivers
- GPU: CUDA, cuDNN, TensorRT, graphics
- camera / video: GStreamer, Argus

## MPS

Here MPS means **CUDA Multi-Process Service**. It lets multiple CUDA processes share the GPU more efficiently. NVIDIA says MPS is a **lightweight runtime service** for cooperative CUDA multi-process workflows, with:

- nvidia-cuda-mps-control
- nvidia-cuda-mps-server
- libcuda.so client runtime

Without MPS, the GPU has to time slice with more context switches:

- process A CUDA context
- process B CUDA context
- process C CUDA context

With MPS:

- process A \
- process B  -> MPS server -> GPU
- process C /

The MPS server **multiplexes** CUDA work from multiple processes into a shared context. That can allow kernels from A/B/C to **overlap** on the GPU when there are enough free GPU resources.

```
Process A launches small kernel using 20% of SMs
Process B launches small kernel using 30% of SMs
Process C launches small kernel using 10% of SMs
```

GPU may run concurrently:

```
time →
A: [kernel AAAAA]
B:   [kernel BBBBBBB]
C:     [kernel CCC]
GPU: overlapping work
```

But if one process launches a big kernel that uses most of the GPU:

```
Process A launches huge RF-DETR kernel using most SMs
Process B launches FoundationPose kernel
```

```
time →
A: [huge kernel AAAAAAAAAAAAA]
B:                         [kernel BBB]
```

So MPS reduces context-switch overhead and enables overlap, but it does not magically make one GPU become three GPUs.

### Runtime Checks

You can check mps with:

```
which nvidia-cuda-mps-control
```

And start it:

```
nvidia-cuda-mps-control -d
```
