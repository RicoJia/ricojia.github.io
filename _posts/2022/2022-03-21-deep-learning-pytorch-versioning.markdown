---
layout: post
title: Deep Learning - PyTorch Versioning
date: '2022-03-21 13:19'
subtitle: 
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

Takeaways:
    - `x.add_()/multiply_()` is to do in-place addition, and updates the gradient.
    - `x+something` actually creates a new tensor.
    - `detach()` means detaching from the computational graph, and creates a new tensor that shares the same data but does NOT require gradients. So if you need to modify the tensor but do not need to modify the gradients, this is one option.
    - `x.clone()` creates a new tensor

Test Code

```python
import torch

# Initial setup
# see version = 0
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"Initial x: {x}")
print(f"Initial x._version: {x._version}")

# out-of-place operation (create a new tensor)
y = x * 2
print(f"Computed y = x * 2: {y}")
# see version = 0
print(f"x._version after y = x * 2: {x._version}")

# out-of-place operation (create a new tensor)
x = x + 1.0
print(f"Modified x after x + 1.0: {x}")
# see version = 0
print(f"x._version after non-in-place operation: {x._version}")

# In-place modification on a detached version of x
x_detached = x.detach()
x_detached.add_(1.0)
print(f"Modified x_detached after x_detached.add_(1.0): {x_detached}")
# see version = 1
print(f"x_detached._version after in-place operation: {x_detached._version}")

x.add_(1)
# See x._version after in-place operation: 2
print(f"x._version after in-place operation: {x._version}")

x_clone = x.clone()
```


