---
layout: post
title: Deep Learning - PyTorch Basics
date: '2022-01-28 13:19'
subtitle: Neural Network Model Components, Common Operations
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---


## Neural Network Model Components

- Softmax layer:

```python
import torch
import torch.nn as nn

# Example softmax layer
softmax_layer = nn.Softmax(dim=1)

# Example input (logits)
logits = torch.tensor([[2.0, 1.0, 0.1],
                       [1.0, 3.0, 0.1]])

# Apply the softmax layer
softmax_output = softmax_layer(logits)
print(softmax_output)
```

- Focal Loss: `nn.softmax()` and `Tensor.gather()`

```python
import torch

# (m, class_num, h, w)
model_preds = torch.Tensor([[
    # two channels
    [[0.1, 0.4, 0.2]],
    [[0.3, 0.6, 0.7]],
]])

# label (m, h, w), only 1 correct class
targets = torch.Tensor([
    [[0, 1, 0]]
]).long()
probs = torch.nn.functional.softmax(inputs, dim=1)
# See
# tensor([[[[0.4502, 0.4502, 0.3775]],

#          [[0.5498, 0.5498, 0.6225]]]])
print(probs)
# See
# tensor([[[[0.4502, 0.5498, 0.3775]]]])
probs.gather(1, targets.unsqueeze(1))
```

- `softmax` creates a softmax across these two channels.
- `tensor.gather(dim, indices)` here will select the softmax values at the locations indicated in targets. `targets` cleverly stores indices of one-hot vecotr as class labels.

## Common Operations

### Convertions Between An Numpy Array And Its Torch Tensor

```python
torch_tensor = torch.from_numpy(np_array)
# convert from float64 to float32
torch_tensor_float32 = torch_tensor.float()

# Create the tensor on CUDA
torch_tensor_gpu = torch_tensor.to('cuda')
# This is production-friendly
torch_tensor = torch_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
# Explicitly creating a cpu based tensor
tensor = tensor.cpu()

# back to numpy array:
np_array = tensor.detach().numpy() 
```

- If an `np_array` is of `float64`, then to convert it to other datatypes using `torch_tensor.float()`

### Reshaping

```python
import torch
tensor_a = torch.randn(2, 3, 4)  # Tensor with some shape
tensor_b = torch.randn(6, 4)     # Another tensor with a different shape

# Reshape tensor_b to match the shape of tensor_a
reshaped_tensor = tensor_b.reshape(tensor_a.shape)
```

- Checking for unique values:

```python
print(torch.unique(target))
```

### Misc

- Printing full tensors

```python
torch.set_printoptions(profile="full")  # Set print options to 'full'
print(predicted_test)
```
