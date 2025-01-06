---
layout: post
title: Deep Learning - Bert 
date: '2022-04-10 13:19'
subtitle: 
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Masked Autoencoder (MAE)

The main idea of the Masked Autoencoder (MAE) is to mask parts of an input (e.g., image features) and train a model to reconstruct the original input by leveraging the remaining unmasked information.

In BERT, random words are masked in a sentence, and the unmasked words are fed into an encoder. The encoder learns contextual representations that help predict the masked words. Similarly, MAE adapts this principle to images but incorporates a feature extraction process before masking.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/a2835a05-6878-4032-afac-cdaaffbc1a43" height="300" alt=""/>
    </figure>
</p>
</div>

Steps in a Masked Autoencoder Workflow:

```
Feature Extraction -> Masking and Encoding -> Embedding Reconstruction -> Decoding and Output
```

- A feature extractor like VGG or another convolutional neural network processes the input image to produce feature maps. These feature maps represent the image in a compressed, high-dimensional space.
- Some features are masked (set to zero or removed), and only the unmasked features are fed into an encoder. The encoder outputs embeddings that match the size of the unmasked features, encoding contextual information about the input.
- The embeddings are reinserted into their corresponding positions within the unmasked feature map to create a full feature map. This reconstructed feature map is fed into a decoder.
- The decoder processes the reconstructed feature map to generate output features with the same size as the original feature map. These features are then passed through a transformation (or directly used) to reconstruct the original image.
