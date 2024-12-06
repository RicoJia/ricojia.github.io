---
layout: post
title: Deep Learning - Optimizations Part 1
date: '2022-01-18 13:19'
subtitle: Momentum, RMSProp, Adam, Learning Rate Decay, Local Minima, Gradient Clipping
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction

Deep learning is still highly empirical, it works well in big data where there's a lot of data, but its theories are not set in stone (at least yet). So use below optimization techniques with a grain of salt - these are common techniques that could generalize well, but they may not be well generalizable to your applications.

## Exponentially Weighted Averages

Exponentially weighted averages is $v_{t} = b v_{t-1} + (1-b)\theta_t$. As $b$ becomes lower, this average will become more noisy. E.g., when $b=0.9$

$$
\begin{gather*}
v_{100} = 0.1 * 0.9^{99} \theta_1 + 0.1 * 0.9^{98} \theta_2 ...
\end{gather*}
$$

In exponentially weighted averages, the initial phase could be much lower than the "real average" values. This is called a bias. And this is because initially, $(1-b) \theta_t$ is the main component, while $(1-b)$ could be small Usually, people won't care much. But if you are concerned, you can then apply bias correction:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/adf23e10-a382-4fb7-90ab-5c8377cbfe2d" height="300" alt=""/>
    </figure>
</p>
</div>

$$
v_t = \frac{v_t}{1-b^t}
$$

As $t \rightarrow \inf$, $v_t$ will go to 1, so the exponentially weighted average will grow closer and closer to the uncorrected one.

## Technique 2 - Gradient Descent With Momentum

When gradient is low, it might be helpful to use the weight itself as part of the momentum to amplify the gradient. So momentum is defined as:

$$
\begin{gather*}
V_{dw} = \beta V_{dw} + (1-\beta) dw
\\
V_{db} = \beta V_{db} + (1-\beta) db
\\
W = W - \lambda V_{dw} \\
b = b - \lambda V_{db}
\end{gather*}
$$

- $\beta$ is commonly 0.9
- No bias correction in momentum implementation

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/956a851f-32b9-4371-8372-ccf12837d310" height="300" alt=""/>
    </figure>
</p>
</div>

- (1) is gradient descent, (2) is with a small momentum, (3) is with a larger momentum

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/204376e7-4532-49fe-b398-e88b1c441d67" height="300" alt=""/>
        <figcaption>Dashed Line Is Gradient, Solid Line Is The Actual dW</figcaption>
    </figure>
</p>
</div>

## Technique 3 - RMSProp (root mean squared Prop)

When gradient has large oscillations, we might want to smooth it out. In the below illustration, gradient is large along W, but small along b. So to smooth out the magnitudes of W, we can apply "RMSProp". (Propsed by Hinton in a Coursera course, lol)

![Screenshot from 2024-08-09 16-15-35](https://github.com/user-attachments/assets/14f79831-653e-47fc-865a-cc0b43730eba)

Mathematically, it is

$$
\begin{gather*}
S_w = \beta S_{dw} + (1-\beta) dW^2 \\
S_b = \beta S_{db} + (1-\beta) db^2 \\
S_w^{corrected} = \frac{S_w}{1-\beta^t} \\
S_b^{corrected} = \frac{S_b}{1-\beta^t} \\
W = W - \lambda \frac{dW}{\sqrt{S_w^{corrected}} + \epsilon} \\
b = b - \lambda \frac{db}{\sqrt{S_b^{corrected}} + \epsilon}
\end{gather*}
$$

- $dW^2$ are element wise squares. So we have root, and squares. (but no "mean"??)
- In real life, to avoid numerical issues when $dW^2$ is small, we want to add a small number $\epsilon$ (typically $10^{-8}$).
- "RMSProp" is used when Adam is not used.
- Typical values:
  - Learning rate: $[10^(-3), 10^{-2}]$
  - Weight decay: $[10^{-5}, 10^{-4}]$

### PyTorch Implementation

[In PyTorch](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html), momentum $\mu B$ and weight decay on the gradients $\lambda_w \cdot W$ are also incorporated. For the sake of conciseness, I'm omitting calculation for bias.

$$
\begin{gather*}
dW = dW + \lambda_w \cdot W
\\
S_w = \beta S_{dw} + (1-\beta) dW^2 \\
S_w^{corrected} = \frac{S_w}{1-\beta^t} \\
B = \mu B + \frac{dW}{\sqrt{S_w^{corrected}} + \epsilon}
\\
W = W - \lambda B \\
\end{gather*}
$$

Note that

- Weight decay $dW = dW + \lambda_w \cdot W$ is finally applied on parameters. This is equivalent to **L2 regularization** which eventually adds an $-\lambda_w W$ term in weight update

$$
\begin{gather*}
L = L(W) + \frac{\lambda_w}{2} W^2
\\
=> L' = L(W)' + \lambda_w W
\\
=> W = W - \lambda \cdot [gradients] \cdot L'
\end{gather*}
$$

## Technique 4 - Adam (Adaptive Momentum Estimation)

Adam combines the RMSProp and momentum all together. For each weight update, we calculate add momentum to the weight, optionally correct it, then divide it by the sum of squared weights. Mathematically, it is:

$$
\begin{gather*}
\text{momentum}
\\
V_{dw} = \beta_1 V_{dw} + (1-\beta_1) dw \\
V_{db} = \beta_1 V_{db} + (1-\beta_1) db \\
\text{RMS Prop}
\\
S_w = \beta_2 S_w + (1-\beta_2)(dW)^2 \\
S_b = \beta_2 S_b + (1-\beta_2)(db)^2 \\

\text{Weight Update}
\\
W = W - \lambda \frac{V_{dw}}{\sqrt{S_w + \epsilon}} \\
b = b - \lambda \frac{V_{dw}}{\sqrt{S_b + \epsilon}}  \\

\end{gather*}
$$

- Additionally, $V_{dw}$, $V_{db}$, $S_w$, $S_b$ can be in their "corrected" form.

- **Hyperparameter Choices**:
  - $\lambda$
  - $\beta_1$ (~0.9)
  - $\beta_2$ (~0.999)
  - $\epsilon$ (~10^{-8})

Usually we don't need to change them. Momentum variable is more probably valuable than the rest.

## Technique 5 - Learning Rate Decay

One less commonly used technique is to decay learning rate.

$$
\begin{gather*}
\alpha = \frac{\alpha}{1 + \text{decay\_rate} * \text{epoch\_num}}
\end{gather*}
$$

If this makes the decay rate too fast, one can use "scheduled learning rate decay":

$$
\begin{gather*}
\alpha = \frac{\alpha}{1 + \text{decay\_rate} * int(\frac{\text{epoch\_num}}{\text{time\_interval}})}
\end{gather*}
$$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/fbdae564-0df5-4fcc-b9a1-3b7d80719d62" height="300" alt=""/>
    </figure>
</p>
</div>

### PyTorch Learning Rate Adjustments

- `torch.optim.lr_scheduler.ReduceLROnPlateau`: reduces learning rate when loss doesn't improve
- `torch.optim.lr_scheduler.StepLR`: reduces learning rate on fixed schedule
- `torch.optim.lr_scheduler.ExponentialL` or `torch.optim.lr_scheduler.CosineAnnealingLR` reduces learning rate in a smooth exp or cosine manner.

## Local Optima

When people think about `gradient=0` in optimizations, local minima immediately becomes a concern, like the ones below.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/eb474c3a-1e64-41dc-9f5f-daf685a88132" height="300" alt=""/>
    </figure>
</p>
</div>

However, in a high-dimensional space, the opportunity to encounter a local minima is very low. Instead, we are more likely to hit a saddle point like the one below (with Andrew Ng's Horse)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/f853fe3c-7afc-41a4-955b-34aa9ae0b984" height="300" alt=""/>
    </figure>
</p>
</div>

So, momentum, Adam are really helpful for getting off the plateaus.

## Experiment Results

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/fa8ace64-1ae1-45f4-9c54-f620734a747e" height="300" alt=""/>
        <figcaption>Cost of SGD</figcaption>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/ea252cc9-ff87-4ada-8d49-1fc4d2be4624" height="300" alt=""/>
        <figcaption>Cost of SGD with Momentum</figcaption>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/02b45a88-33d1-4aaf-9bfb-9fb3ea80fb70" height="300" alt=""/>
        <figcaption>SGD with Adam</figcaption>
    </figure>
</p>
</div>

Momentum helps, but when the learning rate is low, it doesn't create a big of a difference. Adam has really shown its power: there must have been some dimensions that oscillate relatively more intensively than others. On simple datasets, all three could converge to good results. But Adam would converge faster.

Advantages of Adam:

- Relatively low memory requirements (higher than SGD and SGD with momentum)
- Works well with little hyperparameter tuning (except alpha.)

[Adam Paper](https://arxiv.org/pdf/1412.6980)

But If we add **learning rate decay** on top of SGD and SGD with momentum, those methods can achieve similar performances as with Adam:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/6c3acf35-cc0c-43cd-9307-4db21bd337b0" height="300" alt=""/>
        <figcaption>SGD with Momentum and Learning Rate Decay</figcaption>
    </figure>
</p>
</div>

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

### Gradient Clipping Code

When training a transformer for translation, there could be gradient explosion. To veritfy it, one can use

```python
for name, param in (model.named_parameters()):
    # grad is a matrix
    if torch.isnan(param.grad).any():
        print("inf: ", name)
    elif torch.isinf(param.grad).any():
        print("nan: ", name)
# print param sum
print(sum(param.sum() for param in model.parameters() if param.requires_grad))
```

[To prevent gradient norm from going to infinite, gradient clipping can be applied. This is for mixed-precision training](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping)

```python
scaler = GradScaler()
for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # Without unscale_(), unscaling from float16 to float32 happens here
        scaler.step(optimizer)
        # Adjusts the scaling factor for the next iteration. If gradients are too low, increase the scaling factor.
        scaler.update()
```

This is gradient for `float32` training

```python
for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
```
