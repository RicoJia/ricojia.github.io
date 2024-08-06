---
layout: post
title: Deep Learning - Common Issues
date: '2022-01-16 13:19'
subtitle: Regularization
comments: true
tags:
    - Deep Learning
---

## Overfitting

Capacity is the ability to fit a wide variety of functions. Models with complex patterns may also be overfitting, thus have smaller capacity.

### Technique 1: Regulation

Regularization is to reduce overfitting by penalizing the "complexity" of the model. Common methods include:

- L1 and L2 regularization:
    - L1 encourages sparsity: **NOT SUPER COMMON** $\lambda \sum_j || w_j ||$
    - L2 penalizes large weights: $ \lambda \sum_j || w_j^2 ||$. $b$ could be omitted. $\lambda$ is another parameter to tune (regularization parameter)
    - The regularization term is a.k.a "weight decay"

Effectively, some neurons' weight will be reduced, so hopefully, it will result in a simpler model that could perform better on the test set landscape.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/d832260c-662a-41aa-8140-4b4c99b77753" height="300" alt=""/>
        <figcaption>Source: Andrew Ng's Deep Learning Class on Coursera</figcaption>
    </figure>
</p>
</div>

### Technique 2: Dropout

Drop out is to force a fraction of neurons to zero during each iteration. Redundant representation

- At each epoch, **randomly select** neurons to turn on. Say you want 80% of the neurons to be kept. This means we will not rely on certain features. Instead, we shuffle that focus, which spreads the weights
- **VERY IMPORTANT**: **computer vision uses this a lot. Because you have a lot of pixels, relying on every single pixel could be overfitting.**

- So, there are fewer neurons effectively in the model, hence makes the decision boundary simpler and more linear.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e9404d42-64ee-40f7-a949-e140db824006" height="300" alt=""/>
    </figure>
</p>
</div>

During Inferencing, there is no need to turn on drop-out. The reason being, it will add random noise to the final result. You can choose to run your solution multiple times with drop out, but it's not efficient, and the result will be similar to that without drop-out.

But be careful with $J$ visualization, it becomes wonky because of the added randomness.

### Technique 3: Tanh Activation Function

Note: when the activation is tanh, when w is small, the intermediate output z of the neuron is small. tanh is linear near zero. So, the output model is more like a perceptron network, which learns linear decision boundaries. Hence, the model's decision boundary is likely to be more linear. Usually, overfitting would happen when the decision boudary is non linear.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/63808479-fa3c-4cbb-a5c4-7691587e5e06" height="300" alt=""/>
        <figcaption>Source: Andrew Ng's Deep Learning Class on Coursera</figcaption>
    </figure>
</p>
</div>

### Technique 4: data augmentation

One can get a reflection of a cat, add random distortions, rotations,

### Technique 5: Early Stopping

Stop training as you validate on the dev set. If you know realize that your training error is coming back up, use the best one.

**One thing to notice is "orthogonalization"**: the tools for optimizing J and overfitting should be separate.