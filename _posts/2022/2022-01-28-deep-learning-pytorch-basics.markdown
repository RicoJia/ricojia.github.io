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