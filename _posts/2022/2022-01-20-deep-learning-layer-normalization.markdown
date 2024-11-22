---
layout: post
title: Deep Learning - Layer Normalization
date: '2022-01-20 13:19'
subtitle: Normalization For Sequential Data
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Layer Normalization

Batch normalization has two main constraints:

- When batch size become smaller, it performs bad? Nowadays, we tend to have higher data resolution, especially in large NLP training.
- Need to maintain running means. Batch normalization cannot be used on time sequence data
  - Because it operates on the 2nd dimension (channels). In sequence data, that is the time dimension. Since we have variable time length, normalizing across time dimension is feasible, but suffers the variability of number of elements.
- In distributed training, BN needs to combine average and var from multiple devices

Layer normalization mitigates such issues (**hence it's always used in NLP tasks in place of batch normalization**). Also, layer normalization does **NOT require storing running mean and variances** The way it works is to take the mean and variance cross a batches (not cross channels)

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/49b335b5-35bb-477b-8b20-8ab59d638c13" height="200" alt=""/>
       </figure>
    </p>
</div>

### Implementation

```python
import torch
class LayerNormCustom(torch.nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True) -> None:
        """
        Args:
            normalized_shape : a single embedding / picture's shape
            elementwise_affine : unlike batch norm that applies 
                the affine transformation over a channel, layer norm applies element_wise
        """
        super().__init__()
        self.epsilon = 1e-8
        self.elementwise_affine = elementwise_affine
        # (-3, -2, -1)
        self.dims = tuple(range(-len(normalized_shape), 0))
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape)) # So covers all dims except for batches
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape)) # So covers all dims except for batches

    def forward(self, X):
        """
        Args:
            X: For text data, shape is (B, T, C), or (Batch size, Time step, Channels)
            For image data (which is less common), it's (B, C, H, W)
        """
        # mean = X.sum(-1, keepdim=True)/C 
        mean = X.mean(dim = self.dims, keepdim=True)
        # (X - mean)**2.sum(-1, keepdim=True) / C
        var = X.var(dim = self.dims, keepdim=True, unbiased=False)  # unbiased=False for biased norm
        z_norm = (X-mean)/torch.sqrt(var + self.epsilon)
        if self.elementwise_affine:
            z_norm = self.weight * z_norm + self.bias
        # Per-sample normalization?
        return z_norm
```

Layernorm is commonly used for text data. In that case, we do not include the time steps dimension. The implementation yields the same output as torch:

```python
# B, T, C
shape = batch_size, timesteps, channels = 2, 3, 4
X = torch.randn(*shape)
bn = LayerNormCustom(normalized_shape=(channels,))
X_norm = bn(X)
print(f'Rico x norm: {X_norm}')

m = torch.nn.LayerNorm(normalized_shape=(channels,))
X_torch_norm = m(X)
print(f'Torch x norm: {X_torch_norm}')

bn.eval()
X_norm = bn(X)
print(f'============= eval================')
print(f'x norm: {X_norm}')

X_torch_norm = m(X)
print(f'Torch x norm: {X_torch_norm}')
```

For visual transformers (ViT), an image is divided into patches (equivalent to tokens), which then are transformed into feature vectors (just like text data). So the construct is the same as the above.

### Misc Notes Of Layer Normalization

- Can layer norm used on RNN? From the above, yes. All time steps across a specific batch will be normalized.

- Layer normalization was proposed by [Ba et al. 2016](https://arxiv.org/abs/1607.06450) and was incorporated into the Transformer in [Vaswani et al.'s famous paper Attention Is All You Need](https://arxiv.org/abs/1706.03762). GPT-2 took this architecture, but moved the layer normalization to the front of every block. This way, the residual connection of the Transformer is kept clean.

- [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) does not give all the necessary implementation details. Its implementation is buried under 30 layers of auto-generated CUDA code, behind an instructable dynamical dispatcher. This is because PyTorch really really cares about efficiency, fair enough.

- [Andrej Karpathy wrote a very good tutorial on this](https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md)
TODO: Try doing backward prop. See Karpathy's tutorial.