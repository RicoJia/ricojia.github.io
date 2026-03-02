---
layout: post
title: "[ML] Point Cloud Transformer"
date: 2025-02-02 13:19
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
