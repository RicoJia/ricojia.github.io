---
layout: post
title: PyTorch Mixed Precision Training
date: 2026-01-17 13:19
subtitle: torch.zeros
comments: true
tags:
  - CUDA
---
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
