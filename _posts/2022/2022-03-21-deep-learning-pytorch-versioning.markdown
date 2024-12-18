---
layout: post
title: Deep Learning - PyTorch Versioning And Memory Allocation
date: '2022-03-21 13:19'
subtitle: In-Place and Out-of_Place Matrix Ops, Gradient Checkpointing
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## PyTorch Versioning Is Necessary Because We Have In-Place and Out-of_Place Matrix Ops

Takeaways:
    - `x.add_()/multiply_()` is to do in-place addition, and updates the gradient.
    - `x+something` actually creates a new tensor.
    - `detach()` means detaching from the computational graph, and creates a new tensor that shares the same data but does NOT require gradients. So if you need to modify the tensor but do not need to modify the gradients, this is one option.
    - `x.clone()` creates a new tensor

Test Code

```python
import torch

# Initial setup
# see version = 0
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"Initial x: {x}")
print(f"Initial x._version: {x._version}")

# out-of-place operation (create a new tensor)
y = x * 2
print(f"Computed y = x * 2: {y}")
# see version = 0
print(f"x._version after y = x * 2: {x._version}")

# out-of-place operation (create a new tensor)
x = x + 1.0
print(f"Modified x after x + 1.0: {x}")
# see version = 0
print(f"x._version after non-in-place operation: {x._version}")

# In-place modification on a detached version of x
x_detached = x.detach()
x_detached.add_(1.0)
print(f"Modified x_detached after x_detached.add_(1.0): {x_detached}")
# see version = 1
print(f"x_detached._version after in-place operation: {x_detached._version}")

x.add_(1)
# See x._version after in-place operation: 2
print(f"x._version after in-place operation: {x._version}")

x_clone = x.clone()
```

## PyTorch Allocates A Caching Allocator, `torch.cuda.empty_cache()` Clears It

1. When creating tensors on a GPU, PyTorch requests a chunk (e.g., 20MB) larger than the tensor (3MB). The cache really decreases the number of memory request calls.
2. When a tensor on GPU goes out of scope, the memory of the stale tensor remains in PyTorch cache, but it's labelled as `unused`.
3. To free up that cache, we call `torch.cuda.empty_cache()`. Note that **`torch.cuda.empty_cache()` only removes UNUSED caches**

```python
import torch

def print_memory_allocated_and_reserved():
    print(f"Allocated {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Reserved {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
# Check initial GPU memory usage
print("Initial GPU memory:")
print_memory_allocated_and_reserved()

# Allocate a large tensor
a = torch.randn(1000, 1000, device="cuda")
# Check GPU memory usage after allocation
print("After allocating tensor 'a':")
print_memory_allocated_and_reserved()

# Delete the tensor
del a
# Check GPU memory usage after deletion
print("After deleting tensor 'a':")
print_memory_allocated_and_reserved()

# Empty the cache
torch.cuda.empty_cache()

# Check GPU memory usage after emptying cache
print("After emptying cache:")
print_memory_allocated_and_reserved()
```

See:

```python
Initial GPU memory:
Allocated 0.00 MB
Reserved 0.00 MB
After allocating tensor 'a':
Allocated 3.81 MB
Reserved 20.00 MB
After deleting tensor 'a':
Allocated 0.00 MB
Reserved 20.00 MB
After emptying cache:
Allocated 0.00 MB
Reserved 0.00 MB
```

### Allow Memory Segments In CUDA

`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
    - memory is allocated by default contiguously. By using this, we are able to use fragemented memory, but it could be slightly & negligibly slower.

## Gradient Checkpointing

Gradient Checkpointing trades computational overhead for reduced memory consumption. When this is enabled, some intermediate outputs of layers are NOT stored. They will be recomputed in the backward pass (a.k.a checkpointed) when gradients are needed. One illustrative example of what it does is:

```python
y = f3(f2(f1(x)))
f3 = 2x
f2 = z + 3
f1 = 4w
```

With checkpointing, we store `z = 2x`, compute but do not store `w=z + 3`. Eventually, we get output `4w`. But in backpropagation, we compute `w=z + 3` again.
