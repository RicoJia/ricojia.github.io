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
        - Without setting the seed, ` tf.random.normal` would raise `RuntimeError`, but Python and Numpy won't

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

## Mixed Precision Training

Using `float 16` for training

- Needs channel last format `NHWC`

```python
model = model.to(memory_format = torch.channels_last)
input_tensor = input.to(memory_format=torch.channels_last)
output = model(input_tensor)
```