---
layout: post
title: "[CUDA - 1]  GPU Architecture"
date: 2026-01-11 13:19
subtitle: GPU Architecture, Tensor Cores, Pinned Memory
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---
[A Great introduction video can be found here](https://www.youtube.com/watch?v=h9Z4oGN89MU&t=494s)

## GPU (GA102) Architecture

A graphics card's brain is its GPU. NVidia's Ampere GPU architecture family has GA102 and GA104. GA102 is shared across NVidia 3080 (Sep1 2020), 2080Ti (Sep1 2020), 3090 (May 31 2021), 3090Ti (Jan 27 2022). Here's the architecture

```
1 GPU -> 7 2D array of Graphics Processing Clusters (GPC) -> 12 Stream Processors (SM) -> 4 Warps, 1 Ray-Tracing(RT cores), *4 "Special Function Unit"*
```

In a warp:

```
1 warp -> 32 CUDA cores + 1 Tensor Core
```

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/f25cd14e-1081-4e8b-be7c-632bb29d4ff1" height="700" alt=""/>
        <figcaption><a href=""> GA102 Architecture </a></figcaption>
    </figure>
</p>
</div>

The architecture has 10752 CUDA cores, 336 Tensor cores, and 84 RT cores. These cores do all the computation of a GPU

- A CUDA core executes standard floating-point and integer operations. Great for vector calculation, or rastererization in graphics. It's also a "thread"
  - Non-mixed-precision **matrix multiplication and acummulation (MMA)** is done here. Within 1 clock cycle, one multiply and one add is done here.`A x B + C`
  - A different section of the core can do: Bit shifting, bit masking, inqueing incoming operands and instructions, and accumulating outputs.
  - Division, square root, and trigonometry are done on the special function unit in Stream Manager.
  - In ML, this is useful in:
    - scalar operations, summations,
    - data augmentation
    - Activations like ReLU, Softmax
- Tensor Cores: see below "Tensor Cores" section
- A Ray-Tracing corecomputes ray tracing, which simulates how light interacts with surfaces to create realistic shadows and reflections. It handles computations for ray-object intersection and bounding volume (3D Rendering)

The GPU series is manufactured very smart way. Defects usually are damaged cores that affect its own SM circuitry. So depending on a chip's defects, the chip's SM circuitry can be deactivated accordingly so the chip can go on to a 3080, 3080Ti, etc.

- Note that each GPU tier has a different clock speed, which follows the same defect-recycling philosophy as well

### Tensor Cores

[This nice tutorial explains how tensor cores works.](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/) The main features include:

- A tensor core performs mixed-precision **matrix multiplication and acummulation (MMA)** with FP16 and INT8. It's great for ML training.
  - The Ampere architecture can also handle **TF32 float** (a 19-bit floating point value), or **BF16**
- A tensor core breaks down matrix multiplication and addition into 4x4 submatrices, **or tiles** to do `D = A*B + C`. A and B are FP16, C and D are FP16 or FP32.
  - Now, you might be wondering how to break down matrices of any size to this. Let's walk through a simple example.
    - Imagine `A: 10x8, B: 8x4, C: 10x4`. So result `D: 10x4`. Let's allocate that memory
    - Pad A so it's `16x8`. Now A is

            ```
            A11, A12
            A21, A22
            A31, A32
            A41, A42
            ```

    - So A*B is:

            ```
            A11 * B11 + A12 * B12
            A21 * B11 + A22 * B12
            A31 * B11 + A32 * B12
            A41 * B11 + A42 * B12
            ```

    - So, the warp will figure out the synchronization points to finally add results together. The point is with padding, **matrix multiplication of any sizes can be decomposed into tile multiplications and multiple additions**.

### Thread Architecture

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/050d2142-ac95-4ad5-8534-5c2896c36540" height="300" alt=""/>
    </figure>
</p>
</div>

- A warp has 32 threads.
- **Same instructions are issued to the same threads in a warp**
- A thread block or a warp can be organized as **1D or 2D** for indexing. Each thread ultimately, **has 1D indexing**. In general, CUDA's 2D structures follow the below convention (which is different than the pixel convention)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/703cccd7-5122-4c1f-a7d5-04cca28cd9b9" height="200" alt=""/>
    </figure>
</p>
</div>

```cpp
block_id = block_x + block_y * grid_dim.x
thread_id = block_id * (block_dim.x * block_dim.y) + thread_y * block_dim.x + thread_x
```

## What SM Does

GPUs are massively parallel, but there's a huge difference between global memory access vs arithmetic instructions:

- global memory access: 400-800 clock cycles
- arithmetic instructions: 4 cycles

So when a warp does

```cpp
float x = glpbal_memory[i]
```

That warp must wait hundreds of cycles for the data. So an SM will pick another ready warp and execute it. This is **hardware-warp-scheduling**. There's no OS context switch, no overhead, the SM just switches **to a few other warps** to issue instruction for the next cycle.  So **CUDA parallelism is not about running all threads simultaneously, but about always having enough warps to issue instructions to while others wait**

## GPU Peripherals

Voltage Regulation:

- A graphics card will have 12v input, and 1.1v output to the GA102 chip. This could produce significant amount of heat, so we need a heat sink.
- A graphics card also has 24 graphics memory chips (GDDR6x, GDDR7). In gaming, 3D models are loaded  `SSD ->  graphics memory -> L2 cache`.
  -  L2 is just a hardware cache between global memory and the SMs.
  - It's not directly controllable by us
  - A GA102 chip has 2 L2 Cache (3MB each), which is very limited
  - The 24 graphics memory chips transfer 384 bits/s (bus width) for`graphics memory -> L2 cache`. The bandwidth is 1.15Tbytes/s
  - Graphics memory chips micron GDDR6x encodes 2 bits into 1 4-voltage level bit, using PAM4 encoding. But now, the industry agrees to use PAM3, an encoding scheme that uses 3 voltage levels that's on GDDR7
    - This improves power efficiency, reduces encoder complexity, and increases SNR
- High-Bandwidth Memory Chip Micron HBM3E
  - This chip is a memory "cube", with layers of memory that are connected by "Through-Silicon Vias".
  - This is usually used in AI Datacenters 😊

## Data Transfer

### `pin_memory`

A page of memory is the smallest fixed-length block of virtual memory managed by OS. [Please check here for more about paging](../2018/2018-01-20-linux-operating-systems.markdown). In CUDA, tensors are loaded from CPU (host) to GPU (device) following this process:

1. CUDA allocates an array of "temporary page-locked/pinned" memory on CPU as a staging area
2. Copy tensors to the staging area
3. Transfer data from CPU to GPU

So, in Python, `DataLoader(pin_memory=True)` stores samples in page-locked memory and speeds-up the transfer CPU and GPU.

From the [NVidia documentation](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/#pinned_host_memory)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/f6814dd5-cb4c-40d4-b3d7-2bcc3f5ccefb" height="300" alt=""/>
    </figure>
</p>
</div>

- Global memory - either DRAM or VRAM
  - Your batch data live here too.
  - Your tensors, model weights, activations all live here.
  - RTX 30xx use VRAM (GDDR6), lower tier integrated GPU (AMD Radeon Graphics, Intel UHD) **has no dedicated memory, so it carves out a chunk of CPU DRAM. It's slower and lower bandwidth**

---

## Discussions

### What happens if I run two ML inference workloads on one GPU?  

*(e.g., image compression + point cloud compression)*  
  
1. First, you’re far more likely to run out of **VRAM** than compute. If you have, say, an 8GB GPU, that memory must hold:  
  
- Model weights (both models)  
- Activations for the current batch  
- Temporary buffers (e.g., KNN scratch space, attention tensors)  
- CUDA context + allocator caches  
- Possibly optimizer states (if training, though you mentioned inference)

Even in inference-only mode, activations and intermediate buffers can be large.  Point cloud models especially (e.g., KNN graphs, Chamfer distance buffers) can allocate sizable temporary tensors.  
  
If both models together exceed available VRAM:  

- You’ll get **CUDA OOM errors**  
- Or heavy allocator fragmentation  
- Or forced fallback to smaller batch sizes  
  
2. Second, you probably won’t “run out of compute” — you’ll slow down . GPUs don’t crash when two workloads compete for compute. Instead:  
  
- CUDA kernels from both workloads get **time-sliced**  
- SMs (streaming multiprocessors) are shared  
- Kernel launches interleave  
  
The result:  

- Lower throughput per model  
- Increased latency  
- Less predictable performance  
  
Compute is elastic. Memory is not.

2. Read / Write Memory bandwidth contention can hurt a lot. That bandwidth is shared by:  

- All SMs  
- All running kernels  
- All processes using the GPU  
  