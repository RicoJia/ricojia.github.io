---
layout: post
title: Deep Learning - Deeplab Series Theories
date: '2022-08-01 13:19'
subtitle: Deeplab
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Motivation

UNet has **restricted receptive fields**. It is sufficient for identifying local areas such as tumors (in medical images). When we learn larger image patches, UNet was not sufficient.

DeepLab V1 - V3 enlarges its receptive fields by introducing "dilated convolution" (空洞卷积)
DeepLab V2 TODO
DeepLab V3 (2017)
DeepLab v3+ (2018) - Achieved SOTA on Pascal-VOC 2012.

## Receptive Field

- Is finding an receptive field an "upconvolution?"
- Dilated Convolution:
  - Each element in the dilated conv kernel now takes a value in a subregion. This can effectively increase the area the kernel sees, since in most pictures, a small subregion's pixels are likely similar
    ![Convolution_arithmetic_-_Dilation](https://github.com/user-attachments/assets/e14d5e18-6eb2-4cff-9d03-97ad9240988e)

  - An interesting question is: why don't we use a larger kernel? Since VGG networks (2014), using a 3x3 kernel has become a trend.
    - A `7x7` is more than 5 times larger than a `3x3` kernel.
    - A `3x3` introduces more non-linearity than a `7x7`

## SPP (Spatial Pyramid Pooling) Layer

- In Deep learning, "spatial dimensions" means "height and width"
- Some networks need fixed input size for their fully connected layers. When does that happen?
  - Models like VGGNet, AlexNet, and early versions of ResNet were designed with fixed input sizes in mind
- So, to be able to use the dense layers with varying image size, we want to "downsize" feature maps to a fixed size. Spatial Pyramid Pooling Layer creates an pyramid of pooling results on feature maps. E.g.,
    1. Adjust pooling window size so we get a 4x4 pooled results from all input feature maps
    2. Adjust pooling window size so we get a 2x2 pooled results from all input feature maps
    3. Adjust pooling window size so we get a 1x1 pooled results from all input feature maps (global pooling)
    4. Flatten all pooled results, then concatenate them together

![The-structure-of-the-spatial-pyramid-pooling-layer-the-final-feature-vector-for](https://github.com/user-attachments/assets/dea5e37d-ca30-4e42-a672-4f0cc0a97420)

Greatest advantage of SPP is that it can preserve features on different granularity levels.
