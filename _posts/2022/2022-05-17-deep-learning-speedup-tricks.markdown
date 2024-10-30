---
layout: post
title: Deep Learning - Speedup Tricks
date: '2022-05-17 13:19'
subtitle: Op Determinisim, Torch Optimizer Tricks, Mixed Precision Training
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## General Speed-Up Tricks

- If you look to use albumentations for augmentation, sticking to the `[batch, H, W, Channels]` (channel last) could make data loading faster

- `tensor.contiguous()` creates a new tensor that uses contiguous blocks of memory. You might need this after `permute()`, `view()`, `transpose()`, where the underlying memory is not contiguous.

- `torch.cuda.empty_cache()` empties cached variables on GPU. So please do this **before sending anything to the GPU**

- `@torch.inference_mode()` turns off autograd overhead and can be used for evaluate. There's no gradient tensor whatsoever. Accordingly, tensors created under this mode are **immutable**. This is great for **model evaluation**

```python
@torch.inference_mode()
def evaluate_model(input_data):
    output = model(input_data)
    return output
```

    - `torch.no_grad():` might still have intermediate tensors that carry gradients, and **allows for operations that modify tensors in place.**

## Op Determinisim

Here is [a good reference on Op Determinism](https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/config/experimental/enable_op_determinism). Below is how this story goes

- Tensor operations are not necessarily deterministic:
  - `tf.nn.softmax_cross_entropy_with_logits` (From a quick search, it's still not clear to me why this is non-deterministic. Mathematically, the quantity should be deterministic.)
- Op Determinisim will make sure you get the same output with the same code, same hardware. But it will disable asynchronicity, so **it will slow down these operations**
  - Use the same software environment in every run (OS, checkpoints, version of CUDA and TensorFlow, environmental variables, etc). Note that determinism is not guaranteed across different versions of TensorFlow.
- How to enable Op Determinism?
  - PyTorch
  - TensorFlow:

        ```python
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()
        ```

    - This effectively sets the pseudorandom number generators (PRNGs) in  Python seed, the NumPy seed, and the TensorFlow seed.
    - Without setting the seed, `tf.random.normal` would raise `RuntimeError`, but Python and Numpy won't

## Torch Optimizer Tricks

### `RMSProp`

`foreach` option updates weights in a vectorized, batched manner, based on gradients and moving averages like momentum. This is more efficient than python `for loops` and uses optimized linear algebra libraries, like `BLAS`, `cuBLAS`. Without `foreach`, the vanilla RMSProp would be:

```python
# each param is the weight tensor of a layer
for param in model.parameters():
    # Compute gradient g_i
    g_i = param.grad

    # Update running average of squared gradients
    s_i = alpha * s_i + (1 - alpha) * (g_i ** 2)

    # Update parameter
    param -= eta * g_i / (sqrt(s_i) + epsilon)
```

But with `foreach`, the GPU can calculate the stacked weights altogether with stacked gradients and running averages

```
[
  W^{(1)} (matrix of size m x n),
  b^{(1)} (vector of size n),
  W^{(2)} (matrix of size n x p),
  b^{(2)} (vector of size p)
]
```

Caveat:

- Slight numerical differences

### Cache Clearing And Memory Management

- Empty cache

```python
# Tensors are immediately cleared in GPU memory. By default it's not
torch.cuda.empty_cache()
# This resets the internal memory counter that tracks the peak memory usage on the GPU.
# After resetting, you can accurately track peak memory
torch.cuda.reset_max_memory_allocated()
# ensures that all preceding GPU operations have been completed before moving to the next operation.
torch.cuda.synchronize()
```

- Be cautious with operations like `.item()`, `.numpy()`, `.cpu()` as they can cause synchronization.**So move data to the CPU last**

    ```python
    # Not doing item() here because that's an implicit synchronization call
    # .cpu(), .numpy() have synchronization calls, too
    local_correct = (predicted_test == labels_test).sum()
    ```

  - `.sum()` is **not** calling the GPU
