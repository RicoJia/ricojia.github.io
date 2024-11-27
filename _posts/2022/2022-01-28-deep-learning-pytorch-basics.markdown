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

## Data Type Conversions

### Common Data Types

- `torch.arange(start, stop, step)` can take either float or int values
  - `torch.range(start, stop, step)` is deprecated because its signature is different from that of python's `range()`

- `torch.tensor(int, dtype=torch.float32)`. We can't pass an int right into `torch.sqrt()`. We must transform it into a tensor.
  - Note, we are using the function `torch.tensor()`, not the class `torch.Tensor()`

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

### Float to Bool

- If an `np_array` is of `float64`, then to convert it to other datatypes using `torch_tensor.float()`
- Datatypes: if we need to convert a matrix into `int` when seeing errors like `"bitwise_and_cuda" not implemented for 'Float'`, we can do `matrix.bool()`

```python
predicted_test = torch.where(outputs_test > 0.4, 1, 0).bool() 
local_correct = (predicted_test & labels_test).sum().item()
```

### Device Related

- It's crucial to add `.to(X.device)` to a custom model

- In places like `torch.autocast(device_type=device_type, dtype=torch.float16)`, we need to pass a string in.
  - Solution: `device_type = str(device)`

## Neural Network Model Components

- Make a conv-batch-relu module that optionally have components

```python
layers = [nn.Conv2d(), nn.Conv2d() ...]
layers.append(component)    # if necessary
nn.Sequential(*layers)
```

    - `nn.Sequential()` is a sequential container that takes in modules. It has a `forward()` function, and it will pass it on to the first module, then the chain starts.

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

### Math Operations

- `torch.bmm(input, mat2)`: Batch-Matrix-Multiplication
  - If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.

        ```
        outi​=inputi​@mat2i
        ```
- `tensor.numel()` calculates the total number of elements. Returns `batch_size * height * width`.
- `torch.manual_seed(42)` set a seed in the RPNG for both CPU and CUDA.
- `torch.var(unbiased=False)` this is to calculate [biased variance](../2017/2017-06-03-stats-basic-recap.markdown). It's useful in batch norm calculation.
- `torch.Tensor()`'s singleton dimensions are the dimensions with only 1 element.
- `tensor.transpose(-2, -1)`: can transpose a matrix.

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

- There's no difference between `tensor.size()` and `tensor.shape`

### Broadcasting

- dimensions with `1` can be expanded implicitly through broadcasting. E.g., for matrix addition `(3, 1, 4) + (1, 2, 4) = (3, 2, 4)`. Here is how it works:

```
[
  [[a, b, c, d]],
  [[e, f, g, h]],
  [[i, j, k, l]]
]
+
[
  [
    [m, n, o, p],
    [q, r, s, t]
  ]
]
= 
[
  [
    [a, b, c, d] + [m, n, o, p],
    [a, b, c, d] + [q, r, s, t]
  ],
  [
    [e, f, g, h] + [m, n, o, p],
    [e, f, g, h] + [q, r, s, t]
  ],
  [
    [i, j, k, l] + [m, n, o, p],
    [i, j, k, l] + [q, r, s, t]
  ],
  []
]
```

## Misc

- Printing full tensors

```python
torch.set_printoptions(profile="full")  # Set print options to 'full'
print(predicted_test)
```

- Model summary: there are two methods
  - `model = print(model)  # Your model definition`
  - `pip install torchsummary`

        ```python
        from torchsummary import summary
        summary(model, input_size=(channels, height, width))
        ```

## Advanced Topics

### In a custom module, write code for training mode and eval mode

```python
class MyDummy(torch.nn.Module):
    def forward(self):
        if self.training:
            ...
```

### To make variables learnable parameters

```python
class MyModule(torch.nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.my_param = torch.nn.Parameter(torch.Tensor([1,2,3]))
    def forward(self, x):
        return x*self.my_param

m = MyModule()
print(m.my_param, m.my_param.requires_grad, m.my_param.data)
m(torch.Tensor([1,2,3]))
```

torch.nn.Parameter is to represent a learnable param in a neural network.

- `torch.nn.Parameter` autoregisters a parameter in the module's parameter list. It will then be used for computational graph building for
gradient descent.

### Register Buffers

A buffer:
    - is NOT a parameter in a module, so it cannot be learned, and no gradient is computed for them.
    - A buffer's value can be loaded with the module's dictionary. So a buffer can persist between runs.

```python
self.register_buffer('running_mean', torch.zeros(num_features))
```
