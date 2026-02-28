---
layout: post
title: "[ML] Libraries For Point Cloud Compression"
date: 2025-02-09 13:19
subtitle: einops
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---

### Normalization

- `nn.GroupNorm(ngroups, hidden)` splits hidden channels (layers) into `ngroups`, then nomalize points across each group. This could be more stable in small batches. E.g.,  `hidden = 8` channels, `ngroups = 2` . Then channels are split: Group 0: channels 0–3, Group 1: channels 4–7.

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

- Then apply the learned shift and scale:

$$
y = \gamma \hat{x} + \beta
$$

- `nn.ReLU(inplace=True)` modifies the input tensor in place

```python
relu(x) = max(0, x)
```
  - Sometimes it could break autograd if the original pre-Relu value is needed, e.g., skip connections. If those values are not reused they are usually fine.

- **`torch.gather()`**
  - `idx[:, None]` is the same as `idx.unsqueeze(-1)`, which adds a new dimension. For example, an array of shape `(5,)` becomes `(5, 1)`.
  - `idx.repeat(1, fdim, 1)`: copies the tensor `fdim` times along the second dimension. E.g., a `(5, 1)` array after `repeat(1, 3)` becomes `(5, 3)`:

  ```python
  arr = np.array([1, 2, 3, 4, 5])
  arr[:, None].shape        # (5, 1)
  t = torch.tensor(arr[:, None])
  t.repeat(1, 3)            # shape (5, 3)
  ```

  - `torch.gather(input, dim, index)` picks input values by their indices along `dim`:

```python
# - Channel 0 (c=0): `[10, 11, 12, 13, 14]`  
# - Channel 1 (c=1): `[20, 21, 22, 23, 24]`

xyzs =  
[  
  [  # batch 0  
    [10, 11, 12, 13, 14],   # channel 0  
    [20, 21, 22, 23, 24],   # channel 1  
  ]  
]  # shape [1, 2, 5]

# Now choose indices
idx = [[0, 3, 4]]   # shape [1, 3]

torch.gather(xyzs, dim=2, index=idx.unsqueeze(1).expand(-1, 2, -1))
# gives
[  
  [  
    [10, 13, 14],  
    [20, 23, 24],  
  ]  
]   # shape [1, 2, 3]
```

- negative infinity

```python
# Use dtype-aware very negative value  
neg_inf = torch.finfo(logits.dtype).min
```

### Sum

- weighted sum across two dims if their dimension numbers are the same:

$$
\sum_k attn[b, c, m, k] \cdot t[b, c, k, m]
$$

```python
attn = torch.tensor(
[[[[0.2, 0.3, 0.5],  
[0.1, 0.6, 0.3]]]]) # (1,1,2,3) bcmk  
  
t = torch.tensor(
[[[[10., 20.],  
[11., 21.],  
[12., 22.]]]]) # (1,1,3,2) bckm

res = torch.einsum('bcmk, bckm -> bcm', attn, t)
#[0.2 * 10 + 0.3 * 11 + 0.5 * 12, 0.1 * 20 + 0.6 * 21 + 0.3 * 22]
print(res)  # [[[11.3, 21.2]]]
```

- Histogram Count:
 	- `idx`: idx to update
 	- `src`: values to add to counts
 	- `counts`: histogram counting bins

```python
sample_num = 5
idx = torch.tensor([0, 2, 2, 4, 2, 0, 1, 4])   # indices to add into (length N)
src = torch.tensor([3, 22, 1, 1, 1, 1, 1, 1], dtype=torch.long)

counts = torch.zeros(sample_num, dtype=torch.long)
counts.scatter_add_(0, idx, src)

# idx[0] = 0, so counts[0] += src[0] = 3
# idx[1] = 2, so counts[2] += src[1] = 22
# idx[2] = 2, so counts[2] += src[2] = 23
# idx[3] = 4, so counts[4] += src[3] = 1
# idx[4] = 2, so counts[2] += src[4] = 24
# idx[5] = 0, so counts[0] += src[5] = 4
# idx[6] = 1, so counts[1] += src[6] = 1
# idx[7] = 4, so counts[4] += src[7] = 2
print("counts:", counts.tolist())  # [4, 1, 24, 0, 2]
```
