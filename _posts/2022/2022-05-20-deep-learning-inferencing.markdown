---
layout: post
title: Deep Learning - Inferencing
date: '2022-05-20 13:19'
subtitle: Autograd Profiler
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Autograd Profiler

PyTorch's Autograd Profiler provides information on the resources (CPU and GPU) for each operation in a model.

```python
import torch.autograd.profiler as profiler

with profiler.profile(use_cuda=True) as prof:
    # Your inference code here
    outputs = model(dummy_input)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

- `use_cuda=True` enables event tracing
- `prof.key_averages()` prints a table of TIME consumed

```
---------------------------------  ---------------  ---------------  ---------------
Name                               CPU total %      CUDA total %      # of Calls
---------------------------------  ---------------  ---------------  ---------------
aten::mm                             30.00%           45.00%              10
aten::relu                           10.00%           15.00%              10
aten::addmm                          5.00%            8.00%              10
...
---------------------------------  ---------------  ---------------  ---------------
```
