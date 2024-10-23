---
layout: post
title: Deep Learning - PyTorch Model Training
date: '2022-03-06 13:19'
subtitle: Checkpointing
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Checkpointing

Checkpointing is a technique to trade compute for memory during training. Instead of storing all intermediate activations (outputs layers) for backprop, which consumes a lot of memory, checkpointing discards some and recomputes them during the backward pass.  Thus, this saves memory at the expense of additional computation

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.inc = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Example layer
        self.inc = checkpoint.checkpoint(self.inc)  # Enable checkpointing

    def forward(self, x):
        x = self.inc(x)  # Checkpointed layer
        return x
```

checkpointing can be used on functions as well.

## Training

- `model.n_channels       print(f'model.n_channels: {} ')`
