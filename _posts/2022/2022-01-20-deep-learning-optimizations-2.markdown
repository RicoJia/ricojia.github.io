---
layout: post
title: Deep Learning - Optimizations Part 2
date: '2022-01-20 13:19'
subtitle: Batch Normalization (BN), Gradient Clipping
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Batch Normalization

Normalization with mean and variance looks like:
<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/a3e885cd-b6d4-4014-bf8c-8cd6f7890a3b" height="300" alt=""/>
        <figcaption><a href="https://e2eml.school/batch_normalization">Source: Brandon Rohrer  </a></figcaption>
       </figure>
    </p>
</div>

The main idea is: If normalization is effective with input data to improve over learning, why can't we do that during the training? We have seen that we normalize the input data based on their average and mean. Given an input `(N, C, H, W)`, we can normalize across the `C` channel to achieve uniform results. The steps are:

1. Similar to input normalization, we could get input to each layer to have mean 0, and unit variance across all dimensions. Note the addition of $\epsilon$.

$$
\begin{gather*}
\mu = \frac{\sum x}{m}
\\
\sigma = \frac{\sum (x - \mu)^2}{m}
\\
z_{norm} = \frac{x-\mu_z}{\sqrt{\beta^2 + \epsilon}}
\end{gather*}
$$

2. But there are cases where we might want them to have different distributions, actually. Some might want to transform $\tilde{z}$ to a different learnable distribution. Scale(gamma) and shift (beta) are added. This is called an **affine transform**. It's used in TensorFlow and PyTorch by default.

$$
\begin{gather*}
\tilde{z} = \frac{x_i - \mu}{\sigma} \gamma + \beta
\end{gather*}
$$

3. After this transformation, $\tilde{z}$ is fed into the next layer. **Actually in the deep learning community, there is some debate on whether to do BN before or after activation function. But doing BN before the activation function is a lot more common.** Additionally, `Dropout` after BN is more preferred. This was introduced by Sergei Ioffe and Christian Szegedy in 2015.

```
x -> Batch Normalization -> Activation (ReLu, etc.)
```

4. As a result, the distribution of Z of each dimension across the batch is more normalized. So visually,

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/15d7f9d7-13fd-4353-bd8c-eaffdeb85269" height="200" alt=""/>
        <figcaption><a href="https://github.com/user-attachments/assets/15d7f9d7-13fd-4353-bd8c-eaffdeb85269">Source: Ibrahim Sobh</a></figcaption>
    </figure>
</p>
</div>

5. Here is the effect on gradient magnitude distribution with batch normalization. One can see that with BN, gradient magnitudes span across a larger range quite evenly

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/e71315a7-0a47-4fa9-9553-6dadebc839a0" height="200" alt=""/>
            <figcaption><a href="https://viso.ai/deep-learning/batch-normalization/">Source</a></figcaption>
       </figure>
    </p>
</div>

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

**During inference, since mean and variance are fixed, they can be implemented using a linear layer.**

- $\beta_\mu$, $\beta_v$ are momentum constants.

So in total, a batch normalization layer for one channel has **2 trainable parameters ($\beta_\mu$, $\beta_v$) + 2 non trainable parameters ($\mu_z$, $\sigma_z$) = 4 parameters**

Now one might ask: does the order of mini batches affect the learned mean and variance? The answer is yes, but its effect should be averaged out if the mini batches are randomly shuffled.

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

### Practical Use Notes

- It's quite common to see the pattern `conv -> BN -> ReLu`

## gradient clipping

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
