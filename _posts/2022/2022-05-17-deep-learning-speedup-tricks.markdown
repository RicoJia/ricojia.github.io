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

## Mixed Precision Training

Matrix multiplcation, gradient calculation is faster if done in FP16, but results are stored in FP32 for numerical stability. So that's the need for mixed precision training.  Some ops, like linear layers and convolutions are faster in FP16. Other ops, like reductions, often require the dynamic range of float32?

```
use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
# if False, then this is basically noop
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            # gradients are scaled here?
            output = net(input) # this should be torch.float16
            loss = loss_fn(output, target)  # loss is autocast to torch.float32
        # exits autocast before backward()
        # create scaled gradients
        scaler.scale(loss).backward()
        # First, gradients of optimizer params are unscaled here. Unless nan or inf shows up, optimizer.step() is called

        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
```

Or using FP16 througout without scaling

```python
for epoch in range(0): # 0 epochs, this section is for illustration only
    for input, target in zip(data, targets):
        # Runs the forward pass under ``autocast``.
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = net(input)
            # output is float16 because linear layers ``autocast`` to float16.
            assert output.dtype is torch.float16

            loss = loss_fn(output, target)
            # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
            assert loss.dtype is torch.float32

        # Exits ``autocast`` before backward().
        # Backward passes under ``autocast`` are not recommended.
        # Backward ops run in the same ``dtype`` ``autocast`` chose for corresponding forward ops.
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
```

- Note that `torch.autocast` expects the device type ('cuda' or 'cpu'), without the device index.

- According to this [NVidia page](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html):
  - Each Tensor Core performs `D = A x B + C`, where A, B, C, and D are matrices
  - In practice, higher performance is achieved when A and B dimensions are multiples of `8`
  - **Half precision is about an order of magnitude** (10x) faster than double precision (FP64) and about four times faster than single precision (FP32).
  - **Scaling is quite essential here**. Without scaling, loss would diverge.

- DO NOT USE MIXED_PRECISION TRAINING FOR:
  - Reduction is an operation that makes a tensor smaller along one or more dimensions, such as sum, mean, max, min.
    - This is not sitting well with mixed-precision training, e.g., in a MSE loss function, the mean (a reduction) could have underflow issues.
  - Division is not sitting well with mixed-precision training, either. That's because the denominator could have underflow issues.

- Is it better to use `@torch.inference_mode()` or with `torch.no_grad()`?
  - `torch.inference_mode()` is a newer context manager.
    - It disables not only gradient computation but also the **version tracking** of tensors required by autograd
    - Turning off version tracking can be significant for memory usage.

### Results

I observed at 50% speed up in inference when profiling my UNet model using fp16 for inferencing.
