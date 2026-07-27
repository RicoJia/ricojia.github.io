---
layout: post
title: "[CUDA - 6] CUDA Functions"
date: 2026-01-21 13:19
subtitle: attomicAdd, pragma unroll, python kernels
comments: true
header-img: img/post-bg-alibaba.jpg
tags:
  - CUDA
---

## `atomicAdd`

There’s no such thing as “atomic memory” in CUDA—**atomicity is a property of an operation**, not a memory type. `atomicAdd` is an instruction you apply to a **memory location** (typically in **global memory** or **shared memory**) so that the **read → modify → write** sequence happens as one indivisible action.

### Example: histogram update (race condition vs atomic)

Say every thread looks at a value `x` (0–255) and increments a histogram bin `hist[x]`. Below is a race condition because two threads could hit the same bin

```cpp
__global__ void histo_bad(const unsigned char* data, int N, int* hist) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    if (i < N) {  
        int bin = data[i];  
        hist[bin] = hist[bin] + 1;   // race condition  
    }  
}
```

Instead, one needs to use `atomicAdd` to avoid the race condition

```cpp
__global__ void histo_good(const unsigned char* data, int N, int* hist) {  
 int i = blockIdx.x * blockDim.x + threadIdx.x;  
 if (i < N) {  
  int bin = data[i];  
  atomicAdd(&hist[bin], 1); // read-modify-write is indivisible  
 }  
}
```

## `pragma unroll`

- `#pragma unroll` is to tell NVCC compiler  to expand a for loop directly, without loop counter, or branch. This makes instruction scheduling better. **Note, this is a hint, not a guarantee**. But note that this could increase your code size, and consume more registers which could slow you down. **Another note, it's usually not effective on CPU**

```cpp
for (int i = 0; i < 4; i++) {  
  x += a[i];  
}

-> 

x += a[0];  
x += a[1];  
x += a[2];  
x += a[3];
```

- Also,  `#pragma unroll` is effective only on the very next for loop,

```cpp
#pragma unroll  
for (int i = 0; i < 4; ++i) { ... }  
  
for (int j = 0; j < 4; ++j) { ... }   // not affected
```

## Custom Functions

- `DIVUP` is divide and round up. USed to calculate how many blocks are needed to cover up all m elements.

```cpp
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
// for 1000 elements, I have 256 threads per block. How many blocks do I need
dim3 blocks(DIVUP(1000, 256), b);
```

---

## Python Kernels

```python
@wp.kernel(enable_backward=False) def bilateral_filter_depth_kernel(...): ...
```

Python creates the function object
          ↓
wp.kernel(...) receives that function
          ↓
Warp examines and registers its definition

Warp uses the function’s:

- argument type annotations,
- function body,
- Warp operations such as `wp.exp`,
- array reads and writes,
- and `wp.tid()`

to generate native C++ or CUDA code. Warp kernels require statically typed arguments because they are compiled rather than interpreted like ordinary Python. For example:

```
depth: wp.array(dtype=float, ndim=2)
radius: int
sigmaD: float
```

tells Warp the exact data types expected by the compiled kernel. Conceptually, Warp turns something like:

```python
@wp.kernel
def double_values(x: wp.array(dtype=float)):
    i = wp.tid()
    x[i] *= 2.0
```

into something similar to a CUDA kernel:

```c
__global__ void double_values(float* x)
{
    int i = /* thread index */;
    x[i] *= 2.0f;
}
```

Compilation normally happens on the first launch - the decorator primarily **registers** the kernel. Warp compiles the kernels belonging to that Python module when a kernel in that module is launched for the first time. The generated CPU library or CUDA/PTX code is then cached. This triggers execution:

```python
wp.launch(
    kernel=my_kernel,
    dim=1000,
    inputs=[...],
    device="cuda",
)
```
