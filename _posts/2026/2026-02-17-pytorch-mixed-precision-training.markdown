---
layout: post
title: PyTorch Mixed Precision Training
date: 2026-02-17 13:19
subtitle: torch.zeros, GradScaler, GradCheck
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - CUDA
---

## Pytorch Setup

```python
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

- It's important to have both the forward and the backward passes in `autocast`.

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
  - `torch.inference_mode()` is a newer con\text manager.
    - It disables not only gradient computation but also the **version tracking** of tensors required by autograd
    - Turning off version tracking can be significant for memory usage.

### Results

I observed at 50% speed up in inference when profiling my UNet model using fp16 for inferencing.

---

## GradScaler vs Gradcheck

One Big weakness of FP16 is small gradients easily become 0. GradScaler's job is purely about grad values - it fixes **gradient underflow** during FP16 training by **multiplying the loss by a large number before backward**.

Some operations are compatible with fp16, like `Matmuls`, `Conv`, Elementwise operations. Some operations, like reductions (sum), softmax, norm layers, batchnorm require higher precision or might easily overflow in FP16, so they have to remain fp32. This is handled by `torch.autocast`, which decides which ops run in fp16 vs fp32 during the forward pass. The loss itself is whatever dtype `autocast` leaves the final output in, and you typically call `.float()` before the scalar loss computation.

- E.g., in softmax: `sum(exp(x_i))` could overflow in FP16.

### Why is FP16 faster?

**1. Tensor Core acceleration (GEMM)**

GEMM (General Matrix-Matrix Multiplication, i.e. `A x B = C`) is the core of both forward and backward passes. FP16 GEMMs run on dedicated **Tensor Cores**, which offer much higher throughput than the regular CUDA cores used for FP32.

Backprop through linear/conv layers is also GEMM-heavy:

$$
dX = dY \cdot W^T \qquad dW = X^T \cdot dY
$$

Tensor Cores accelerate these as well.

**2. Memory bandwidth**

- FP16 activations are **half the size** of FP32, so less data moves between HBM and compute units.
- Better cache utilization and faster layer-to-layer movement.
- For many models, training is partly bandwidth-limited, so this helps significantly.

### Workflow of Mixed Precision Training?

```
forward (autocast):
    some ops make autocast cast parameters to fp16, otherwise remain fp32
    ↓
    loss (usually fp32 scalar)

scale:
    scaled_loss = loss * scale   (e.g. ×1024)

scaled_loss.backward():
    → gradients are scaled (larger magnitude)
    → It's conducted FP16/FP32, dtype decided by autocast
    → less likely to underflow in fp16

unscale:
    gradients = gradients / scale

optimizer step:
    fp32 master weights updated using unscaled gradients
    model fp16 weights updated from fp32 master copy
```

In PyTorch, model weights are stored as **fp32**. `Autocast` temporarily casts weights to fp16, then the optimizer updates fp32 weights.
    - FP16 is NOT used in backprop

- `GradScaler` scales all gradients by just scaling the loss $L_{\text{scaled}} = L \times S$, so all gradients in the graph are scaled automatically: $\frac{\partial L_{\text{scaled}}}{\partial \theta} = S \cdot \frac{\partial L}{\partial \theta}$

- Inf/NaN detection — if the scale is too large and causes overflow in the gradients, GradScaler detects it, skips the `optimizer.step()`, and halves the scale. If steps succeed it slowly increases the scale back up

### GradCheck

`Gradcheck` verifies gradient correctness (numerical vs analytical) of custom autograd functions, like CUDA kernels. It requires double precision (FP64), small eps perturbations, so it has nothing to do with underflow above

```
analytical gradient ≈ numerical finite difference gradient
```

It uses Taylor's Theorem to numerically compute gradients via the centered finite difference:

$$
f'(x) \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}
$$

If `epsilon = 1e-6`, `x + 1e-6 == x`, numerator is 0, gradient = 0, mismatch. So **gradcheck require FP64 to work with**

```python
from torch.autograd import gradcheck
```

fp16 underflows to zero for values under `6 x 10^-5`.  Gradients deep in the network are often too small and simply vanish

`gradcheck` perturbs one scalar input element at a time (not “each point vector” together).

---

## `torch.zeros`

By default:

```python
out = torch.zeros(b, c, m).to(features.device)  # allocates on CPU, then transfer to device
```

- `torch.zeros(...)` creates a tensor with dtype **`torch.float32`** unless you specify `dtype=...`.
- `.to(features.device)` moves it to the same device (CPU/GPU) as `features`.

However, it does **not** guarantee the same dtype as `features` if `features` is fp16/bf16/etc. If you want it to match `features` exactly (device + dtype), do:

```cpp
out = torch.zeros((b, c, m), device=features.device, dtype=features.dtype)
```
