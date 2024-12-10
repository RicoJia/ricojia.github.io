---
layout: post
title: Deep Learning - Common Oopsies
date: '2022-05-17 13:19'
subtitle: Underflow
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

### Underflow

- `torch.softmax(X)` X is zero due to underflow.

### Sizing

- Be careful with **the last batch** if you want to initialize any tensor that's specific to each batch's sizes, because it could be smaller than the commonly defined `BATCH_SIZE` since the batch could be truncated.

## Weight Manipulation

### Weight Copying Without `torch.no_grad()`

This is because we are directly updating the parameters. We don't want gradient tracking.

```python
with torch.no_grad():
    in_proj_weight = torch.cat(
        [my_mha.Wq.weight, my_mha.Wk.weight, my_mha.Wv.weight], dim=0
    )
    torch_mha.in_proj_weight.copy_(in_proj_weight)
```
