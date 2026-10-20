---
layout: post
title: Deep Learning - Truncated Normal Distribution
date: 2026-10-16 14:19
subtitle: Rejection Sampling, trunc_normal_, ViT Positional Embedding Initialization
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Deep Learning
  - Machine-Learning
---

A **truncated normal distribution** is a normal distribution restricted to an interval `[a, b]`: values outside it are never produced.

## 1. Rejection Sampling Example

Suppose we draw these eight values from a standard normal distribution, and we only accept values inside `[-1, 1]`:

|Sample|-1.8|0.3|-0.5|1.7|0.9|-0.2|2.1|0.6|
|---|---|---|---|---|---|---|---|---|
|Result|Rejected|Accepted|Accepted|Rejected|Accepted|Accepted|Rejected|Accepted|

If we need eight valid samples, we keep drawing replacements for the three rejected values until we have eight accepted samples. This is called **rejection sampling**, one way to sample from a truncated normal distribution.

## 2. The Density

Mathematically, the resulting probability density is:

$$
f_{\text{trunc}}(x)= \begin{cases} \dfrac{f(x)}{P(-1\le X\le 1)},&-1\le x\le1\\ 0,&\text{otherwise} \end{cases}
$$

where `f(x)` is the original normal density. For the standard normal distribution, `P(−1 ≤ X ≤ 1) ≈ 0.6827`, so within the permitted interval:

$$
f_{\text{trunc}}(x)=\frac{f(x)}{0.6827}\approx1.465\,f(x)
$$

The shape inside the interval is unchanged; the remaining density is just **scaled up by `1 / (allowed CDF area)`** so it still integrates to 1.

Unlike clipping a Gaussian with `torch.clamp`, truncation does not pile rejected values up at the boundaries.

## 3. Code

PyTorch's `nn.init.trunc_normal_` uses truncation bounds `a = -2` and `b = 2` by default, **in the actual units of the weights, not in standard deviations**. With `std=0.02`, those bounds sit at ±100σ, so essentially no sample is ever truncated. To truncate at ±2 standard deviations, set the bounds explicitly:

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

weights = torch.empty(10000)

nn.init.trunc_normal_(
    weights,
    mean=0.0,
    std=0.02,
    a=-0.04,   # -2 * std
    b=0.04     # +2 * std
)

print(weights.min().item())    # >= -0.04
print(weights.max().item())    # <=  0.04
print(weights.mean().item())   # ~0
print(weights.std().item())    # ~0.0176, not 0.02
```

Two details:

- The printed std is smaller than `0.02`: cutting off the tails at ±2σ shrinks the standard deviation to about `0.88σ`. `std` is the parameter of the underlying normal, not the std of the result.
- PyTorch does not actually use rejection sampling. It uses the inverse-CDF method: draw uniformly between `CDF(a)` and `CDF(b)`, then map back through the inverse CDF. The result has the same distribution, without any redraws.

## 4. In ViT

ViT positional embeddings are added to patch embeddings:

$$
\boxed{Z_0=X_{\text{patch}}+E_{\text{pos}}}
$$

If a positional embedding contains unusually large initial values, those values can disproportionately influence the initial transformer activations. The motivation for a truncated normal is to **avoid unusually large values during initialization while preserving random variation**.

Suppose the learnable positional embedding tensor is:

```python
self.position = nn.Parameter(
    torch.zeros(1, 144, 64)
)

nn.init.trunc_normal_(self.position, std=0.02)
```

That is `1 × 144 × 64 = 9216` learnable embedding values. The initialization gives them small, approximately zero-centered random values instead of leaving every value at zero. For example, the first embedding vector might begin as `[0.012, -0.007, 0.018, -0.003, ...]`.

**Subtle but important:** with this exact call, `std=0.02` does **not** mean truncation happens at ±0.04. The default bounds are ±2 in absolute units, so this is effectively a plain `N(0, 0.02²)` initialization. You must explicitly set `a=-0.04, b=0.04` to truncate at ±2σ.

Also, truncated normal is not fundamentally necessary for a ViT. Standard Gaussian initialization can also work. Training stability depends on the entire initialization scheme, normalization, residual connections, learning rate, and architecture.

## 5. Note: Sinusoidal vs. Learned Position Embeddings

Self-attention by itself has no notion of token order, so both methods **inject position information into the tokens** by adding a position vector to each token embedding. They differ in where that vector comes from:

| |Sinusoidal positional encoding|Learned position embedding|
|---|---|---|
|Defined by|A fixed formula of sines and cosines at different frequencies|An `nn.Parameter`, one vector per position|
|Learned?|No|Yes, trained by backprop like any weight|
|Initialization|None needed|e.g. `trunc_normal_(std=0.02)`, as above|
|Used in|The original Transformer ([Transformer Series 1](https://ricojia.github.io/2022/03/26/deep-learning-transformer-1-positional-encoding-and-masking/))|ViT, BERT|

So the truncated-normal initialization above only matters for learned embeddings: sinusoidal encodings are computed, not initialized.
