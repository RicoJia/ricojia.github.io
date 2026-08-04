---
layout: post
title: "[ML] Attention Pooling"
date: 2026-07-25 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - robotics
---
## Attention Pooling

Pooling converts a variable number of vectors into one fixed-size representation. For N point features,

$$  
\mathbf{x}_1,\mathbf{x}_2,\ldots,\mathbf{x}_N,  
$$

max pooling independently keeps the largest value in each feature channel:

$$  
z_j=\max_i x_{i,j}.  
$$

Attention pooling instead learns how much each point should contribute. A query $\mathbf{q}$ is compared with a key $\mathbf{k}_i$ from every point:

$$  
s_i=\frac{\mathbf{q}^{T}\mathbf{k}_i}{\sqrt{d}},  
\qquad  
\alpha_i=\operatorname{softmax}(s_i).  
$$

Where the query q could be from a different source (cross-attention), or from the same input features (self-attention).  Recall that in self attention, Q, K, V come from the same input set X, but they are projected with different matrices:

$$
Q=XW_q​,K=XW_k​,V=XW_v​.
$$

The output is the weighted sum of the corresponding value vectors:

$$  
\mathbf{z}=\sum_{i=1}^{N}\alpha_i\mathbf{v}_i.  
$$

The weights satisfy

$$  
\sum_{i=1}^{N}\alpha_i=1.  
$$

Therefore, the network **can emphasize informative points and suppress less useful ones**. Unlike self-attention, which normally outputs one updated vector per input, **attention pooling reduces the entire set to one vector**.

Attention pooling tends to work well when:

- Different points **have predictably different importance**.

It may perform poorly when:

- Training data is limited.
- Synthetic and real data have different noise or point distributions.
- A simple operation such as max or mean pooling already captures the useful signal.
- Its additional parameters are trained with untuned hyperparameters (Learning rate, Weight decay, dropout, etc.)

Attention pooling is therefore not automatically better than max pooling. It offers more flexible aggregation, but that flexibility helps only when the learned importance pattern transfers to the evaluation data.
