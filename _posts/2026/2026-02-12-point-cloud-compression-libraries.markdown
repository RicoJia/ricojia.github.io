---
layout: post
title: "[ML] Libraries For Point Cloud Compression"
date: 2025-02-12 13:19
subtitle: einops
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---

## einops

Einops is a lightweight Python library that makes tensor reshaping, permutation, tiling, and reduction readable and explicit. It works with PyTorch, NumPy, and TensorFlow.

- **`rearrange`** — replaces `view`/`permute` chains with a readable axis notation:

```python
# Without einops
x = x.view(b, c, h*w).permute(0, 2, 1)

# With einops
x = rearrange(x, 'b c h w -> b (h w) c')
```

- **`repeat`** — adds a new dimension by tiling data along it:

```python
from einops import repeat

# x.shape == (b, n, d)
x = repeat(x, 'b n d -> b n k d', k=4)
# x.shape == (b, n, 4, d)
```
