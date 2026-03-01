---
layout: post
title: "[CUDA - 3] - Mixed Precision Training"
date: 2026-01-16 13:19
subtitle: AT_DISPATCH_FLOATING_TYPES_AND_HALF
comments: true
tags:
  - CUDA
---

## Run Time Type Dispatch

`AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, name, lambda)` expands into something like:

```cpp
switch (points_tensor.scalar_type()) {  
  case at::kFloat: {  
    using scalar_t = float;  
    return [&] { /* your code */ }();  
  }  
  case at::kDouble: {  
    using scalar_t = double;  
    return [&] { /* your code */ }();  
  }  
  case at::kHalf: {  
    using scalar_t = at::Half;  
    return [&] { /* your code */ }();  
  }  
  default:  
    TORCH_CHECK(false, "gathering_forward: unsupported dtype");  
}
```

So `scalar_t` is **not declared by you** in that file — it’s introduced _by the dispatch macro itself_. That’s why `scalar_t` “magically exists” inside the lambda body.

So you can do:

```cpp
void gathering_forward_cuda_launcher(int batch_size, int channels, int n, int m, at::Tensor points_tensor, const int* __restrict__ idx, at::Tensor out_tensor)
{
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(points_tensor.scalar_type(), "gathering_forward",
                                        [&]
                                        {
                                            gathering_forward_cuda_kernel<scalar_t><<<dim3(batch_size, channels, 1), min(ceiling_power_of_2(m), TOTAL_THREADS), 0>>>(
                                                batch_size, channels, n, m, points_tensor.data_ptr<scalar_t>(), idx, out_tensor.data_ptr<scalar_t>());
                                        });
}

template <typename scalar_t>
__global__ void gathering_forward_cuda_kernel(int batch_size, int channels, int n, int m, const scalar_t* __restrict__ points, const int* __restrict__ idx,
                                              scalar_t* __restrict__ out){
                                              
                                              }
                                              
                                              
```

The reason the launchers need to take `at::Tensor` instead of `float*` is that `AT_DISPATCH_FLOATING_TYPES_AND_HALF` must be called somewhere that has the tensor's dtype available.

Before: uses `float*`

```cpp
-void gathering_forward_cuda_launcher(int batch_size, int channels, int n, int m, 
const float * __restrict__ points, const int * __restrict__ idx, float * __restrict__ out);

//  float *grad_points = grad_points_tensor.data_ptr<float>();
```

After: uses at::Tensor

```cpp
void gathering_forward_cuda_launcher(int batch_size, int channels, int n, int m, at::Tensor points_tensor, const int* __restrict__ idx, at::Tensor out_tensor)
```

---

## Atomic Add

**`gpuAtomicAdd` — how it works:**

`atomicAdd` is a CUDA built-in with hardware support for `float`, `double`, `int`, and (on cc ≥ 7.0) `__half2`. It has **no overload for `__half*`** directly.

`gpuAtomicAdd` (from `<ATen/cuda/Atomic.cuh>`) is ATen's wrapper that fills the gap. Internally,

```cpp
// What ATen does internally for Half:
template <>
__device__ void gpuAtomicAdd(c10::Half* addr, c10::Half val) {
    // emulates half atomicAdd using CAS (compare-and-swap) on the 32-bit word
    // containing the half, since hardware only supports __half2 atomics
    unsigned int* base = (unsigned int*)((char*)addr - ((size_t)addr & 2));
    unsigned int old = *base, assumed;
    do {
        assumed = old;
        // patch the right 16-bit half inside the 32-bit word, reinterpret, CAS
        old = atomicCAS(base, assumed, ...);
    } while (assumed != old);
}
```

So for `float` it just calls `atomicAdd` directly (zero overhead). For `half` it falls back to a CAS loop — slower but correct. It's not "safer" in the sense of being free — it's **correctness-safe across all float types** with a perf cost only for half.
