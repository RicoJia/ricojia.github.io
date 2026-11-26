---
layout: post
title: Deep Learning - Contrastive Learning
date: 2026-10-13 13:19
subtitle: InfoNCE Loss, Supervised Contrastive Learning (SupCon)
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Deep Learning
  - Machine-Learning
---

**Supervised contrastive learning (SupCon)** trains a model to produce embeddings in which images of the same class are close together and images of different classes are far apart. Ordinary classification only asks the model to predict the correct label; SupCon also explicitly trains the relationships between feature vectors.

## 1. The Basic Idea

Imagine a dataset of cats and dogs:

| Cat — Image A | Cat — Image B | Dog — Image C | Dog — Image D |
|:---:|:---:|:---:|:---:|
| ![Ginger cat](https://images.openai.com/static-rsc-4/tmaVf3AFe7ljwiPZoL89rm1BQA5ldM4g32xkChtmzWkzFcFJoOB2CPhTGCQlOF1TAhczd-hhtqq19z0Hg7eSvUWlZfdVMWLaYROoH4NCWvQeVgFM9uWjGvwBHInRZwGqGL0oHcJYKHHvadC5BNaM2_MpOWLtbeTnX6WUy-tUgIk?purpose=inline) | ![British cat](https://images.openai.com/static-rsc-4/XUqN-N9ELgNaHuB4wwrTexhDiiUm-S09g4ZXlIKb7LVWKrDfHy46Oi0Ci3lpXNcPXgZ2jKsEgtu0oajDKaJWOIkRGVO9CmLNrMfVomR6fwM7xnkUK067z5VZFBdxAyzqZYqD50PsbFU6dLeflR0htZOo4pJzEnpx2kNDfhZoyJbc676pir61KUggNnKe6cPK?purpose=inline) | ![Golden retriever](https://images.openai.com/static-rsc-4/6WOsZp133T3qN9VvkXSKsOAM893K24QvOkkopYzve_Tjof3SgchzcAJEfz5ReY5fj8iz8VMVXSUGhEzfoRIfHEWHuIcl8V9fy_GRpxe8QgvT0RsuISr6WyF8tZoJYRMdkCxuiOFn88xsIL3q9GPeYpFyZp1lKZIXUczIXBZ9K4Q?purpose=inline) | ![Labrador retriever](https://images.openai.com/static-rsc-4/2krZUsWE0lOtwl2NWIsDP49rhIfbTmgZRaCwlPcUyBlJ5kDPXiRuPgH9Mz4W753P_lW56T0IQJgHm_GA4wQhthMKjSv7RAG5dk1OU4zjjMwq40h8R-8AEEGeHbRvLf1PNgwl3t2DJNtanNkitIGrO1cwwq_U-MBEpxGIaAYCzDg?purpose=inline) |

- An ordinary classifier learns to answer: *is this image a cat or a dog?*
- SupCon adds another objective: make the embeddings of different cats similar to one another, and distinguish them from dog embeddings. For example, the two cats should have similar embeddings even though their colors and appearances differ.

## 2. Training Architecture

A typical image classifier:

![](https://i.postimg.cc/VN80xS6M/mermaid-diagram.png)

SupCon adds a separate projection head and a contrastive loss:

![](https://i.postimg.cc/0ymJXG8n/mermaid-diagram-(1).png)

Both branches share the CNN trunk. During training, cross-entropy teaches the network to classify images, while SupCon encourages the **embeddings** to have meaningful class-based relationships. The projection head is typically a small MLP, and its output is L2-normalized before computing similarities. The projection head can be discarded after training.

## 3. Measuring Similarity

Each image is passed through the CNN and projection head to produce an embedding vector. For two embeddings `z_i` and `z_j`, the cosine similarity is below; if both embeddings are already L2-normalized, it simplifies to a dot product:

$$
s_{ij}=\frac{\mathbf z_i^\top\mathbf z_j}{\|\mathbf z_i\|\|\mathbf z_j\|}
\qquad\xrightarrow{\ \|\mathbf z\|=1\ }\qquad
\boxed{s_{ij}=\mathbf z_i^\top\mathbf z_j}
$$

Cosine similarity ranges from -1 to 1, where 1 means the vectors point in the same direction. Suppose we obtain the following similarities:

|Pair|Similarity|Desired behavior|
|---|---|---|
|A–B (cat–cat)|0.30|Increase|
|A–C (cat–dog)|0.80|Decrease|
|A–D (cat–dog)|0.10|Already relatively low|

The problem is that Image A (cat) is more similar to Image C (dog) than to Image B (cat). Contrastive loss penalizes this arrangement.

## 4. InfoNCE Loss

InfoNCE stands for Info **Noise-Contrastive Estimation**. It's basically a softmax loss that pulls similar (positive) samples together in the feature space, while pushing dissimilar (negative) samples further apart.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/k55tzGW4/download-(5).jpg" height="300" alt="InfoNCE pulls the positive pair together and pushes negatives apart"/>
        <figcaption>InfoNCE pulls the positive pair together and pushes negatives apart</figcaption>
    </figure>
</p>
</div>

It's widely used in CLIP training, supervised contrastive learning (SupCon, below), and temporal contrastive learning, which is SupCon-style training where the "labels" come from when or where samples were taken.

Let's say we have 5 samples: A, B, C, D, E. A and B are similar because they were taken at the same place. C, D, E were taken at a different place. The similarity function `s(·,·)` is cosine similarity.

First, choose an anchor sample, A. There's only 1 positive pair, (A, B). The negatives are (A, C), (A, D), (A, E). The loss is the negative log of the softmax probability of the positive:

$$
\begin{gather*}
\mathcal L_A=-\log\frac{e^{s(A,B)/\tau}}{e^{s(A,B)/\tau}+\sum_{k\in\{C,D,E\}}e^{s(A,k)/\tau}}
\end{gather*}
$$

Splitting the log, then factoring `e^{s(A,B)/τ}` out of the sum:

$$
\begin{gather*}
\mathcal L_A=\log\Big(e^{s(A,B)/\tau}+\sum_k e^{s(A,k)/\tau}\Big)-\frac{s(A,B)}{\tau} \\
=\log\Big(e^{s(A,B)/\tau}\Big(1+\sum_k e^{(s(A,k)-s(A,B))/\tau}\Big)\Big)-\frac{s(A,B)}{\tau} \\
\boxed{\mathcal L_A=\log\Big(1+\sum_k e^{(s(A,k)-s(A,B))/\tau}\Big)}
\end{gather*}
$$

`τ` is the temperature. So you can see, the loss only depends on how much each negative beats the positive, `s(A,k) - s(A,B)`:

- The loss is in `(0, ∞)`, since the sum is always positive.
- The more similar A and B are (relative to the negatives), the smaller the loss.
- The temperature doesn't simply make the loss smaller. As `τ` grows, every exponent goes to 0, so the loss goes to `log(1 + K)` for `K` negatives, and it can no longer tell good embeddings from bad ones. As `τ` shrinks, the hardest negative (the one with the largest `s(A,k)`) dominates the sum.

With the cat/dog similarities from Section 3 (positive B at 0.30, negatives at 0.80 and 0.10), and the same numbers but with the positive raised to 0.90:

| `s(A,B)` | `τ = 0.1` | `τ = 1` | `τ = 100` |
| -------- | --------- | ------- | --------- |
| 0.30     | 5.01      | 1.24    | 1.10      |
| 0.90     | 0.31      | 0.86    | 1.10      |

When a negative beats the positive (first row), a larger `τ` lowers the loss. When the positive already wins (second row), a larger `τ` raises it. Both approach `log 3 ≈ 1.10`. The 5.01 is the same value as the SupCon example below: with one positive, SupCon is exactly InfoNCE.

## 5. Supervised Contrastive Loss

InfoNCE has one positive per anchor. SupCon generalizes it to all same-class images in the batch. One commonly used supervised contrastive loss is:

$$
\boxed{ \mathcal L_i = -\frac{1}{|P(i)|} \sum_{p\in P(i)} \log \frac{\exp(s_{ip}/\tau)} {\sum_{a\ne i}\exp(s_{ia}/\tau)} }
$$

where:

- `i`: the anchor image, i.e. the image currently being compared against the others.
- `P(i)`: the set of **positives**, the other images in the batch with the same class label as `i`. `|P(i)|` is how many there are, so the loss averages over all positives. Each term is the softmax probability of one positive, so the more similar same-class images are, the lower the loss.
- `s_ip`: cosine similarity between the anchor and a positive image.
- `τ`: temperature, controlling how strongly similarity differences affect the loss.
- `a`: any other image in the batch, positive or negative.

For our example, Image A is the anchor, Image B is the only positive, and C and D are negatives. With `τ = 0.1`, the softmax probability assigned to B, and the resulting loss, are:

$$
p_B = \frac{e^{0.30/0.1}} {e^{0.30/0.1}+e^{0.80/0.1}+e^{0.10/0.1}} =\frac{e^3}{e^3+e^8+e^1}\approx0.0067
\qquad\Rightarrow\qquad
\mathcal L_A=-\log(0.0067)\approx5.01
$$

## 6. Combining Classification and Contrastive Losses

The ordinary classification loss is still present:

$$
\boxed{ \mathcal L_{\text{total}} = \mathcal L_{\text{CE}} + \lambda\mathcal L_{\text{SupCon}} }
$$

where:

- `L_CE`: cross-entropy loss for predicting cat versus dog.
- `L_SupCon`: contrastive loss that organizes embeddings by class.
- `λ`: hyperparameter controlling the relative importance of contrastive learning.

The original SupCon paper (Khosla et al., 2020) trains in two stages instead: first the encoder with SupCon alone, then a linear classifier on top of the frozen encoder. Adding the two losses together as above is a common single-stage variant.

## 7. Summary

- In ordinary supervised classification, we train the model to predict the correct class.
- In supervised contrastive learning, we additionally train the model to organize its embedding space by class.
- The network isn't explicitly told where to place each embedding. It learns feature representations through gradient descent on the contrastive loss.
- For cat and dog classification, this means images of different cats produce relatively similar embeddings, while cat and dog embeddings become easier to distinguish.
