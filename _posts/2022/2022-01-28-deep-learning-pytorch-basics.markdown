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
  - Or alternatively, use `math.sqrt()`

- to invert a bool mask: `~key_padding_mask`

- datatype check: `tensor.dtype`, not `type(tensor)`

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

### Data Loading

```python
train_sampler = RandomSampler(train_data)
```

### Make a conv-batch-relu module that optionally have components

```python
layers = [nn.Conv2d(), nn.Conv2d() ...]
layers.append(component)    # if necessary
nn.Sequential(*layers)
```

    - `nn.Sequential()` is a sequential container that takes in modules. It has a `forward()` function, and it will pass it on to the first module, then the chain starts.

- `nn.Sequential()` does not support input args in `seq_layers(X, other_args)`. In that case, use `nn.ModuleList()` and manually iterate through the layers

```python
encoder_layers = torch.nn.ModuleList([
    EncoderLayer(embedding_dim=self.embedding_dim, num_heads=num_heads, dropout_rate=dropout_rate) 
    for _ in range(encoder_layer_num)
])

for encoder_layer in self.encoder_layers:
    X = encoder_layer(X, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
```

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
    - `LazyLinear`  dims are initialized during first pass
    - `optimizer.zero_grad()` should always come before the backward pass
    - ` with torch.autograd.set_detect_anomaly(True):` can be used to print a stack trace
    - indexing: "arr_2d[:, 0] = arr_1d"

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
- Masking

```python
a = torch.ones((4))
mask = torch.tensor([1,0,0,1]).bool()
a = a.masked_fill(mask, float("-inf"))
a   # see tensor([-inf, 1., 1., -inf])
```

- `NaN` comparison: `torch.allclose()` does not handle nan. This is to replace `NaN` with sentinel

```
def allclose_replace_nan(tensor1, tensor2, rtol=1e-05, atol=1e-08, sentinel=0.0):
    tensor1_replaced = torch.where(torch.isnan(tensor1), torch.full_like(tensor1, sentinel), tensor1)
    tensor2_replaced = torch.where(torch.isnan(tensor2), torch.full_like(tensor2, sentinel), tensor2)
    return tensor1_replaced, tensor2_replaced
```

### Reshaping

```python
import torch
tensor_a = torch.randn(2, 3, 4)  # Tensor with some shape
tensor_b = torch.randn(6, 4)     # Another tensor with a different shape

# Reshape tensor_b to match the shape of tensor_a
reshaped_tensor = tensor_b.reshape(tensor_a.shape)
# reshape using -1, which means "inferring size"
tensor_a = torch.randn(2, 3, 4) 
tensor_a.reshape(3, -1).shape   # 3, 8
tensor_b.reshape(-1).shape  # see 24.
```

- `-1` means "inferring size".

- Checking for unique values:

```python
print(torch.unique(target))
```

- There's no difference between `tensor.size()` and `tensor.shape`

- Transpose
  - `tensor.transpose(dim1, dim2)`: swaps the 2 dims in a matrix. Equivalent to applying `permute()` if we only re-arrange two dims there

    ```python
    a.transpose(1, 3)
    ```

  - `tensor.t()` can only transpose a 2D matrix.

    ```python
    a = torch.rand(2, 3)
    print(a.shape)
    print(a.t().shape)
    ```

  - Note that after `transpose()`, the data did not change but the `strides` and `shape` changed. `Tensor.view()` requires contiguous data, so before `view()` one needs to call `contiguous()`.

    ```python
    a1 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32)
    ```

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
    - torchsummary only supports passing an input tensor of float() into the model, then trace the model structure
- It's better to use `torchinfo`a package, [due to this issue](https://discuss.pytorch.org/t/issue-in-printing-model-summary-due-to-attributeerror-tuple-object-has-no-attribute-size/116155/4)
  - `summary(model, input_size, batch_dim=batch_dim)`
    - `input_size` should NOT contain batch size. `torchinfo` will unsqueeze it in batch_dim.
    - One can pass in input data directly as well.

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

- Is NOT a parameter in a module, so it cannot be learned, and **no gradient is computed for them.**
- A buffer's value can be loaded with the module's dictionary. So a buffer can persist between runs.
- Once `model.to()` is called, the register buffer will be moved over as well as part of it.

```python
self.register_buffer('running_mean', torch.zeros(num_features))
```
