---
layout: post
title: Deep Learning - CNN Basics
date: '2022-01-31 13:19'
subtitle: Filters, Convolutions
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Filtering

1. Filters (aka kernels): "Pattern Detectors". Each filter is a small matrix, which you can drag along an image and multiply pixel values with (convolution). They can detect edges, corners, and later, parts of dogs.

When a filter cross correlates with a matrix that has the same shape, it will generate a high response. If it encounters an "opposite" shape, it will generate a negative response.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="" height="300" alt=""/>
        <figcaption><a href="https://github.com/user-attachments/assets/ad49909c-fc10-4d5e-a1c9-4c9275fec19f"> A 45 deg Edge </a></figcaption>
    </figure>
</p>
</div>

## Convolutional Layer

Adding bias: `output[m, o] += bias[o]` bias is added across output channels. For each output channel, **bias is a single number.**

### Benefits Of Using Convolutional Layer

- Reduces the number of parameters, thus reduces the overfitting problem.
- **Smaller number of parameters also mean smaller sets of training images**
- Convolutional Layers also benefit from **sparsity of connections**. This means that the activation of the next layer is only affected by a small number of activations from the previous layer (the ones in the corresponding filtered area)

