---
layout: post
title: Deep Learning - Optimizations Part 2
date: '2022-01-20 13:19'
subtitle: Batch Normalization (BN)
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction

We have seen that we normalize the input data based on their average and mean. We can apply the same with the output of each layer.

```bash
x -> z (normalize here) -> a
```

So similar to input normalization, we could get input to each layer to have mean 0, and unit variance across all dimensions. Note the addition of $\epsilon$.

$$
z_{norm} = \frac{z-\mu_z}{\sqrt{\beta^2 + \epsilon}}
$$

But there might be cases where we might want them to have different distributions, actually. Sowe might want to transform $\tilde{z}$ to a different learnable distribution. With $\gamma$ and $\beta$ being **learnable**:

$$
\tilde{z} = \gamma z_{norm} + \beta = \gamma \frac{z-\mu_z}{\beta} + \beta
$$

After this transformation, $\tilde{z}$ is fed into the next layer. **Actually in the deep learning community, there is some debate on whether to do BN before or after activation function. But doing BN before the activation function is a lot more common.**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/22626f58-cc6d-4999-a36f-ed9638103b8d" height="300" alt=""/>
        <figcaption>Flow Chart Of Batch Normalization With Epsilon Omitted </figcaption>
    </figure>
</p>
</div>

As a result, the distribution of Z of each dimension across the batch is more normalized. So visually,

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/15d7f9d7-13fd-4353-bd8c-eaffdeb85269" height="200" alt=""/>
        <figcaption><a href="https://github.com/user-attachments/assets/15d7f9d7-13fd-4353-bd8c-eaffdeb85269">Source: Ibrahim Sobh</a></figcaption>
    </figure>
</p>
</div>

This was introduced by Sergei Ioffe and Christian Szegedy in 2015.

### During Training

Batch normalization is usually implemented as its own layer. The layer parameters $\gamma$ and $\beta$ can be learned through gradient descent (we can add our own flavors too, like Adam/Momentum/RMSProp). The mean and variance however, comes from the input mini-batch.

One trick is in forprop, when calculating the connected layer before batch normalization, we **don't need to add b** to z. That's because we will be taking the mean of z at all z's dimensions over the batch, and that eliminates b. So, below suffices:

$$
\begin{gather*}
z = W^T x
\end{gather*}
$$

### During Inference

The batch Normalization layer already has its $\gamma$ and $\beta$ learned. The mean and variance are exponentially decayed average learned from training:

$$
\begin{gather*}
\mu_z = \beta_\mu \mu_z + (1-\beta_\mu) \bar{z} \\
\sigma_z = \beta_v \sigma_z + (1-\beta_v) var(z)
\end{gather*}
$$

- $\beta_\mu$, $\beta_v$ are momentum constants.

So in total, a batch normalization layer for one channel has **2 trainable parameters ($\beta_\mu$, $\beta_v$) + 2 non trainable parameters ($\mu_z$, $\sigma_z$) = 4 parameters**

Now one might ask: does the order of mini batches affect the learned mean and variance? The answer is yes, but its effect should be averaged out if the mini batches are randomly shuffled.

## Why Batch Normalization Works?

**Covariate Shift** is the situation where the input data distribution $P(X)$ is shifted, but conditional output distrinbution `P(Y|X)` remains the same. Some examples are:

- In a cat classifier, training data are black cats, but test data are orange cats
- In an image deblurring system, images are brighter than test data.

Batch normalization overcomes the covariate shift in **hidden layers**.

**BN has a slight regularization effect**: Similar to regularization,

- BN reduces sensitivity to weight initialization and learning rate
- BN reduces model overfitting.
  - The network sees somewhat different input distributions in each batch, so this prevents the network from "memorizing" the training data and overfitting.
  - And that adds noise to weight gradient in each mini-batch.
  - BN can reduce the magnitude of outputs, hence the gradients and the weights (**This is regularization**)

However batch normalization doesn't reduce the model complexity so the regularization is very mild.

## Practical Use Notes

- It's quite common to see the pattern `conv -> BN -> ReLu`
