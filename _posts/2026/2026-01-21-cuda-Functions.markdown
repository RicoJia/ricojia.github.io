---
layout: post
title: "[CUDA - 6] SIMT in CUDA"
date: 2026-01-21 13:19
subtitle: attomicAdd
comments: true
tags:
  - CUDA
---
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
