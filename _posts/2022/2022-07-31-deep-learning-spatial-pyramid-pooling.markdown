---
layout: post
title: Deep Learning - Spatial Pyramid Pooling
date: '2022-08-01 13:19'
subtitle: SPP, Ablation Study
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## SPP (Spatial Pyramid Pooling) Layer

In Deep learning, "spatial dimensions" means "height and width". Some networks need fixed input size for their fully connected layers. When does that happen?

- Models like VGGNet, AlexNet, and early versions of ResNet were designed with fixed input sizes in mind
- So previously, we need to crop/warp images.

So, to be able to use the dense layers with varying image sizes, **we want to "downsize" feature maps to a fixed size**. Spatial Pyramid Pooling Layer creates an pyramid of pooling results on feature maps. E.g.,

1. Adjust pooling window size so we get a 4x4 pooled outputs from all 256-d input feature maps
2. Adjust pooling window size so we get a 2x2 pooled outputs from all 256-d input feature maps
3. Adjust pooling window size so we get a 1x1 pooled outputs from all 256-d input feature maps (global pooling)
4. Flatten all pooled outputs, then concatenate them together into an 1-d vector
5. The 1-d vector goes into the FC network as usual.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/dea5e37d-ca30-4e42-a672-4f0cc0a97420" height="300" alt=""/>
       </figure>
    </p>
</div>

The greatest advantage of SPP is that **it can preserve features on different granularity levels**. **This allows training with images of different sizes**.

SPP as a concept is an extension of Bag-of-Words, it was considered in 2014 by He et al (MicroSoft) in;

## Key Insights During Training

Here is an illustration of error rates for multiple models with SPP and multi-size training. As can be seen, SPP and multi-size training can additively increase the model accuracy.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/c5a138cf-3891-425f-8a4d-fd267d35accd" height="300" alt=""/>
            <figcaption><a href="https://medium.com/coinmonks/review-sppnet-1st-runner-up-object-detection-2nd-runner-up-image-classification-in-ilsvrc-906da3753679`">Source: Sik-Ho Tsang </a></figcaption>
       </figure>
    </p>
</div>


## Background Information

- Authors: Kaiming He et al. (Microsoft)
- Competitions: 2nd place ILSVRC Object Detection, 3rd place ILSVRC Classification
- Datasets:

| Dataset         | Task           | Size       | Partitioning                          |
|-----------------|----------------|------------|---------------------------------------|
| ImageNet 2012   | Classification | 1.2M+      | Train: 1.2M, Val: 50K, Test: 100K     |
| Pascal VOC 2007 | Detection      | ~10K       | Train/Val/Test                        |
| Caltech101      | Classification | ~9K        | Train: 30/category, Rest: Test        |

- Comparisons with SOTA models (2014)

| Model                | Year | Classification Accuracy (ImageNet) | Detection mAP (Pascal VOC 2007) | Key Innovations                         |
|----------------------|------|------------------------------------|----------------------------|-----------------------------------------|
| SPP-net              | 2014 | 84.5%                              | 59.2%                       | Spatial pyramid pooling, No fixed-size input |
| AlexNet              | 2012 | 81.8%                              | 58.0%                       | First deep CNN on ImageNet              |
| VGG                  | 2014 | 86.8%                              | 66.3%                       | Deep, uniform 3x3 convolutional layers  |
| GoogLeNet (Inception)| 2014 | 89.2%                              | 62.1%                       | Inception modules, deeper network       |
| R-CNN                | 2014 | -                                  | 53.3%                       | Region proposals, CNN for detection     |
| Overfeat             | 2014 | 81.9%                              | 53.9%                       | Multi-scale sliding windows             |


## Ablation Study

An ablation study in deep learning is to selectively "ablate" (remove) a part of a model, e.g., a layer, an activation function, to study the impact on the overall performance.

## References

[1] [He, K., Zhang, X., Ren, S., & Sun, J. (2015). Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9), 1904–1916.](https://arxiv.org/pdf/1406.4729)
