---
layout: post
title: "[ML] Point Cloud Transformer"
date: 2026-02-02 13:19
subtitle: Vector Attention
header-img: img/post-bg-o.jpg
tags:
  - Machine-Learning
comments: true
---

## Terminology

- modulate
 	- In ML, modulation means *changing one signal using another signal*.
 	- So Attention $output=\alpha \odot v$  with elementwise product is modulation on value features. The value is scaled first before being summed over
- MLP (Multi-Layer-Perceptron)
 	- *A feedforward network made of linear layers + non-linearities*. Technically, we are using Neurons, not just perceptrons (Rosenblatt, 1958)
 	- So `Conv1d + GroupNorm + ReLU + Conv1d` is a valid MLP

## Point Transformer vs Traditional Transformer

## Overview

1. In a point cloud, our input is the raw point cloud. Assuming we have B batches, M points in each batch. P is `[B, 3, M]`. Each point is

$$
p_i = [x_i, y_i, z_i]
$$

2. we first calculate K Nearest Neighbors (KNN) for each point. N is the raw neighbor point set and is `[B, 3, M, K]`

$$
n_i = knn(p_i)
$$

3. Then we apply self attention to each point and their K neighbors. This is to learn the structural features like edges and corners of the point cloud. Each input point $p_i$ is first mapped to an embedding $\mathbf{x}_i \in \mathbb{R}^d$  . The total x tensor is `[B, D, M]`

$$
x_i = MLP(p_i)
$$

4. Then, we feed $x_i$ as the only input to the self-attention unit. linear projections generate query, key, and value features. $x_i$ embedding gets fed into the point transformer network. **So $q_i$, $k_i$, and $v_i$ are learned intermediate feature vectors of a point**. We normally choose q, k, v to have the same dimension, so the total $q, k, v$ tensors are: `[B, H, M]`

$$
\mathbf{q}_i = W_q \mathbf{x}_i, \quad  
\mathbf{k}_i = W_k \mathbf{x}_i, \quad  
\mathbf{v}_i = W_v \mathbf{x}_i.
$$

5. Now, broadcast  each point $p_i$  and its feature vector $q_i$ k times to a new dimension. For the sake of readability, we stick to the notation $p_i$ and $q_i$, so the total `p` is `[B, 3, K, M]`.  $q$ tensor is: `[B, H, K, M]`

$$
\tilde{p}_i =
\begin{bmatrix}
p_i & p_i & \cdots & p_i
\end{bmatrix}_{3 \times K}
\quad
\tilde{q}_i =
\begin{bmatrix}
q_i & q_i & \cdots & q_i
\end{bmatrix}_{H \times K}
$$

6. Calculate positional encoding $\delta_{i,j}$ between **point coordinates $p_i$ and its K neighbors**, so distance and direction information is encoded. $\delta$ is `[B, H, M, K]`:

$$
\delta_{i,j} = MLP(p_i - n_{i,j})
$$

7. For each point $i$, attention is computed over its KNN neighborhood $\mathcal{j}$  (so it's not global). we can calculuate a logit  **which is a vector**. Logit is `[B, H, M, K]`:  

$$
l_{i,j} = MLP(\mathbf{q}_i - \mathbf{k}_j + \delta_{ij})
$$

8. So, for each channel, we can calculate a softmax probability where $\delta_{ij}$ encodes relative positional information and a small MLP creates an additional learnable mapping for the logit to increase expressiveness and stablizes training.

$$
\alpha_{ij} = \mathrm{Softmax}_j \big( MLP(\mathbf{q}_i - \mathbf{k}_j + \delta_{ij}) \big),
$$

9. Then, the output of the attention is **vector attention** that uses elementwise product $a \odot b$, which modulates features in $v_j + \delta_{i,j}$ per channel. Output is `[B, H, M]`:

$$
\mathbf{y}_i' = \sum_{j \in \mathcal{N}(i)}  
\alpha_{ij} \odot \big( \mathbf{v}_j + \delta_{ij} \big),
$$

## Discussions

### What is a Channel?

Think of each channel as detecting something different:

- Channel 3 → curvature
- Channel 7 → edge direction
- Channel 12 → density pattern

Now different neighbors may contribute differently **per channel**. So now, channel 3 might prefer the left neighbor. channel 7 might prefer the above neighbor. channel 12 might prefer the closest neighbor.

Of course, in a real network, each channel could be something more extract.

### Positional Encoding $\delta_{i,j}$ is relative position

Vanilla transformer:

```
x = token embedding + absolute positional encoding
```

Then you do

```
q = Wq x, k = Wk x, v = Wv x
```

In a Point Transformer, position embedding is relative:

$$
\delta_{i,j} = h(p_i - p_j)
\\
\phi(q_i, k_j) = \gamma (q_i - k_j + \delta_{i,j})
$$

### Vector Attention & Elementwise Scalar Product

Recall that a regular attention logit and its probability are a scalar. Given query $q_i \in \mathbb{R}^d$, key $k_j \in \mathbb{R}^d$, and value $v_j \in \mathbb{R}^d$:

$$
\phi_{i,j} = q_i^\top k_j
$$

$$
\alpha_{i,j} = \operatorname{softmax}_j\!\left( \frac{\phi_{i,j}}{\sqrt{d}} \right)  
= \frac{\exp\!\left( \frac{q_i^\top k_j}{\sqrt{d}} \right)}  
{\sum\limits_{l} \exp\!\left( \frac{q_i^\top k_l}{\sqrt{d}} \right)}
$$

Point Transformer (Zhao et al., 2021 ICCV) modified this. Instead of using dot product, the vector attention uses hardamard product

The output is a $C$-dimensional feature vector.  For each channel $c$, the aggregated feature at point $i$ is computed as a weighted sum over its k neighbors:

$$
y_i^{c} = \sum_{k \in \mathcal{N}(i)} \alpha_{ik}^{c} \, v_k^{c},
$$
where $\alpha_{ij}^{c}$ denotes the attention weight assigned to neighbor $j$ for channel $c$, and $v_j^{c}$ is the $c$-th channel of the value feature at point $j$.

**Main benefit for using vector attention is per-channel (where a channel is a feature dimension) weighted sum of attention weight and feature vector**.

Example:

We have 2 channels and 2 neighbors.

$$
V =  
\begin{bmatrix}  
2 & 1 \\  
4 & 2  
\end{bmatrix}  
\in \mathbb{R}^{2 \times 2}
$$

- Rows = channels $C$
- Columns = neighbors $K$

So

$$
\mathbf{v}_1 =  
\begin{bmatrix}  
2 \\  
4  
\end{bmatrix},  
\quad  
\mathbf{v}_2 =  
\begin{bmatrix}  
1 \\  
2  
\end{bmatrix}.
$$

Attention Tensor (Vector Attention): each channel has its own attention distribution over neighbors:
$$
\boldsymbol{\alpha} =  
\begin{bmatrix}  
0.25 & 0.75 \\  
0.90 & 0.10  
\end{bmatrix}  
\in \mathbb{R}^{2 \times 2}
$$

- Row c = attention weights for channel c
- Column k = weight for neighbor k

where row $c$ contains the attention weights for channel $c$, and column $k$ corresponds to neighbor $k$. Vector attention performs an elementwise (Hadamard) product per channel and sums over neighbors:

$$
\mathbf{y} = \sum_{k=1}^{K} \boldsymbol{\alpha}_{:,k} \odot \mathbf{v}_{:,k}.
$$
Equivalently, channel-wise:
$$
y_c = \sum_{k=1}^{K} \alpha_{c k} \, v_{c k}.
$$

Channel 0:

$$
y_0 = 0.25 \cdot 2 + 0.75 \cdot 4 = 3.5  
$$  
  
Channel 1:  

$$y_1 = 0.90 \cdot 10 + 0.10 \cdot 20 = 11  
$$  
Final output vector:
$$
\mathbf{y} =  
\begin{bmatrix}  
3.5 \\  
11  
\end{bmatrix}  
$$
