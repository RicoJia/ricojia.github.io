---
layout: post
title: Deep Learning - Optimizations Part 2
date: '2022-01-20 13:19'
subtitle: Batch Normalization (BN), Gradient Clipping, Layer Normalization
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Batch Normalization

Among many pitfalls of ML, statistical stability is always high on the list. Model training is random: the initialization, even the common optimizers (SGD, Adam, etc.) are stochastic. Because of this, ML models might run into sharp minima in the loss landscape. That would cause high gradients. The most common way to fix it is **batch-normalization**.
Normalization with mean and variance looks like:

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/a3e885cd-b6d4-4014-bf8c-8cd6f7890a3b" height="300" alt=""/>
        <figcaption><a href="https://e2eml.school/batch_normalization">Source: Brandon Rohrer  </a></figcaption>
       </figure>
    </p>
</div>

**The main idea is**: If normalization is effective with input data to improve over learning, why can't we do that during the training? We have seen that we normalize the input data based on their average and mean. Given an input `(N, C, H, W)`, we can normalize across the `C` channel to achieve uniform results. For example, if we have `N` pictures with `HxW` RGB channels, the Batch norm will be:

1. Add up all pixel values across all `N` pictures on the R channel. Then, divide by the sum by `NxHxW`. This will give us one number
1. Repeat the above for G and B channels.

### Overall Model Architecture

```
x -> Batch Normalization -> Activation (ReLu, etc.)
```

- **Actually in the deep learning community, there is some debate on whether to do BN before or after activation function. But doing BN before the activation function is a lot more common.**
- Additionally, `Dropout` after BN is more preferred. This was introduced by Sergei Ioffe and Christian Szegedy in 2015.

### During Training

1. Similar to input normalization, we could get input to each layer to have mean 0, and unit variance across all dimensions. Some variables are:

- $\epsilon$: ensures numerical stability
- $\beta_\mu$, $\beta_v$ are momentum constants.
- In total, a batch normalization layer for one channel has **2 trainable parameters ($\beta_\mu$, $\beta_v$) + 2 non trainable parameters ($\mu_z$, $\sigma_z$) = 4 parameters**

$$
\begin{gather*}
\mu = \frac{\sum x}{m}
\\
\sigma^2 = \frac{\sum (x - \mu)^2}{m}
\\
\mu_r = \beta_\mu * \mu_r+ (1 - \beta_\mu) * \mu
\\
\sigma^2_r= \beta_v * \sigma^2_r + (1 - \beta_v) * \sigma^2
\\

==>
\\

z_{norm} = \frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}}
\\
\tilde{z} = \gamma z_{norm} + \beta
\\
\end{gather*}
$$

- Now one might ask: does the order of mini batches affect the learned mean and variance? The answer is yes, but its effect should be averaged out if the mini batches are randomly shuffled.

#### In forward prop, Skip Adding Bias

That's because we will be taking the mean of z at all z's dimensions over the batch, and that eliminates b. So, below suffices:

$$
\begin{gather*}
z = W^T x
\end{gather*}
$$

### During Inference

The batch Normalization layer already has its $\gamma$ and $\beta$ learned. In training, we simply use the learned $\mu$ and $\sigma$ from training. [In this coursera video](https://www.youtube.com/watch?v=5qefnAek8OA), Andrew Ng stated that this is fairly robust.

$$
\begin{gather*}
z_{norm} = \frac{x-\mu_r}{\sqrt{\sigma_r^2 + \epsilon}}
\\
\tilde{z} = \gamma z_{norm} + \beta
\\
\end{gather*}
$$

- **During inference, since mean and variance are fixed, they can be implemented using a linear layer.**

## Implementation

Now, let's enjoy some code. This is inspired by [ptrblck's implementation](https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py)

```python
import torch
class BatchNormCustom(torch.nn.Module):
    def __init__(self, num_features, affine=True) -> None:
        """
        Scale (gamma) and shift (beta) are learnable parameters for each channel

        Args:
            affine (bool, optional): _description_. Defaults to True.
        """
        # Why in torch, we need to specify num_features in args? Because we need to use it to initialize gamma and beta.
        super().__init__()
        self.affine = affine
        self.epsilon = 1e-8
        self.momentum_factor = 0.9  # this is applied on running mean
        self.gamma = torch.nn.Parameter(torch.ones(num_features).view(1, num_features, 1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(num_features).view(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(num_features).view(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(num_features).view(1, num_features, 1, 1))
    def forward(self, X):
        # We are taking the average across each channel, so the output shape is (num_channel, num_rows, num_clns)
        # While we train, we use the batch mean and var. We just secretly calculate their running averages
        # and use them in eval
        if self.training:
            mean = X.mean(dim = (0, 2, 3), keepdim=True)
            var = X.var(dim = (0, 2, 3), keepdim=True, unbiased=False)  # unbiased=False for biased norm
            # We don't need to track the grads in running_mean and running_var
            with torch.no_grad():
                self.running_mean = self.momentum_factor * self.running_mean + (1-self.momentum_factor) * mean
                self.running_var = self.momentum_factor * self.running_var + (1-self.momentum_factor) * var
        else:
            mean = self.running_mean
            var = self.running_var
        z_norm = (X-mean)/torch.sqrt(var + self.epsilon)
        
        if self.affine:
            z_norm = self.gamma * z_norm + self.beta
        return z_norm
```

### Why Batch Normalization Works?

**Internal Covariate Shift** is the situation where the input data distribution $P(X)$ is shifted, but conditional output distrinbution `P(Y|X)` remains the same. Some examples are:

- In a cat classifier, training data are black cats, but test data are orange cats
- In an image deblurring system, images are brighter than test data.

Batch normalization overcomes the covariate shift in **hidden layers**. **[In this post, Brandon Rohrer cites that BN helps smooth the rugged loss landscape. This allows optimizing with relatively large learning rate](https://e2eml.school/batch_normalization)**

**BN has a slight regularization effect**: Similar to regularization,

- BN reduces sensitivity to weight initialization and learning rate
- BN reduces model overfitting.
  - The network sees somewhat different input distributions in each batch, so this prevents the network from "memorizing" the training data and overfitting.
  - And that adds noise to weight gradient in each mini-batch.
  - BN can reduce the magnitude of outputs, hence the gradients and the weights (**This is regularization**)

However batch normalization doesn't reduce the model complexity so the regularization is very mild.

As a result, the distribution of Z of each dimension across the batch is more normalized. So visually,

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/15d7f9d7-13fd-4353-bd8c-eaffdeb85269" height="200" alt=""/>
        <figcaption><a href="https://github.com/user-attachments/assets/15d7f9d7-13fd-4353-bd8c-eaffdeb85269">Source: Ibrahim Sobh</a></figcaption>
    </figure>
</p>
</div>

Here is the effect on gradient magnitude distribution with batch normalization. One can see that with BN, gradient magnitudes span across a larger range quite evenly

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/e71315a7-0a47-4fa9-9553-6dadebc839a0" height="200" alt=""/>
            <figcaption><a href="https://viso.ai/deep-learning/batch-normalization/">Source</a></figcaption>
       </figure>
    </p>
</div>

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
        var = X.var(dim = (1, 2, 3), keepdim=True, unbiased=False)  # unbiased=False for biased norm
        z_norm = (X-mean)/torch.sqrt(var + self.epsilon)
        if self.elementwise_affine:
            z_norm = self.weight * z_norm + self.bias
        # Per-sample normalization?
        return z_norm
    
batch_size, num_channel, num_rows, num_clns = 2, 3, 4, 5
# X = torch.ones((batch_size, num_channel, num_rows, num_clns))
X = torch.randn(batch_size, num_channel, num_rows, num_clns)
bn = LayerNormCustom(normalized_shape=(num_channel, num_rows, num_clns))
X_norm = bn(X)
print(f'Rico x norm: {X_norm}')

m = torch.nn.LayerNorm(normalized_shape=(num_channel, num_rows, num_clns))
X_torch_norm = m(X)
print(f'Torch x norm: {X_torch_norm}')


bn.eval()
X_norm = bn(X)
print(f'============= eval================')
print(f'x norm: {X_norm}')

X_torch_norm = m(X)
print(f'Torch x norm: {X_torch_norm}')
```

### Misc Notes Of Layer Normalization

- Can layer norm used on RNN? From the above, yes. All time steps across a specific batch will be normalized.

- Layer normalization was proposed by [Ba et al. 2016](https://arxiv.org/abs/1607.06450) and was incorporated into the Transformer in [Vaswani et al.'s famous paper Attention Is All You Need](https://arxiv.org/abs/1706.03762). GPT-2 took this architecture, but moved the layer normalization to the front of every block. This way, the residual connection of the Transformer is kept clean.

- [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) does not give all the necessary implementation details. Its implementation is buried under 30 layers of auto-generated CUDA code, behind an instructable dynamical dispatcher. This is because PyTorch really really cares about efficiency, fair enough.

- [Andrej Karpathy wrote a very good tutorial on this](https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md)
TODO: Try doing backward prop. See Karpathy's tutorial.

## Gradient Clipping

One simple method is: if gradint surpasses a simple threshold, we clip the gradient to the threshold.

- `np.clip(a, a_min, a_max, out=None)`: `out` is an output array

```
# np.clip(a, a_min, a_max, out=None)
np.clip(a, 1, 8)
array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])`
```

The effect of gradient clipping is

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/cba1cc6f-8033-4dad-aa50-5122fb9fc320" height="300" alt=""/>
    </figure>
</p>
</div>
