---
layout: post
title: Deep Learning - Deeplab Series Theories
date: '2022-08-01 13:19'
subtitle: Deeplab, ASPP
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Motivation

UNet has **restricted receptive fields**. It is sufficient for identifying local areas such as tumors (in medical images). When we learn larger image patches, UNet was not sufficient.

DeepLab V1 & V2 (2016) [1]: they were reviewed together as they both use Atrous convolution, or "dilated convolution" (空洞卷积), and **Fully Connected Conditional Random Field**. V1 was only using VGG16, while V2 was using VGG16 and ResNet.
DeepLab V3 (2017) [2]
DeepLab v3+ (2018) - Achieved SOTA on Pascal-VOC 2012.

## The Use of Dilated Convolution

- Some notes about [Dilated Convolution is here](../2017/2017-01-07-transpose-convolution.markdown)

- An interesting question is: why don't we use a larger kernel? Since VGG networks (2014), using a 3x3 kernel has become a trend. That's because:
    - A `7x7` is more than 5 times larger than a `3x3` kernel.
    - A `3x3` introduces more non-linearity than a `7x7`, which allows the network to learn more complex landscapes.
- Fun fact: after 2014, `3x3` seems to become the most popular kernel size

## ASPP

TODO

## References

[1] [He, K., Zhang, X., Ren, S., & Sun, J. (2015). Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(9), 1904–1916.](https://arxiv.org/pdf/1406.4729)

[2] [Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2016). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4), 834–848. https://doi.org/10.1109/TPAMI.2017.2699184](https://arxiv.org/pdf/1606.00915)

[3] [Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv preprint arXiv:1706.05587.](https://arxiv.org/pdf/1706.05587) 

[3] [Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. Proceedings of the European Conference on Computer Vision (ECCV), 801–818.](https://arxiv.org/pdf/1802.02611)

