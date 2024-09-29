---
layout: post
title: Deep Learning - Image Segmentation
date: '2022-02-13 13:19'
subtitle: Encoder-Decoder, Fully-Convolutional Networks (FCN), U-Net
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Pre-Requitstes

### Encoder-Decoder framework

Autoencoders (or encoders) and autodecoders (or decoders) were introduced in the late 1980. An autoencoder compresses input data into smaller dimensions, and a decoder reconstructs them back into their original dimensions. Encoder-decoder is often useful for dimensionality reduction and feature learning.

Originally used for denoising and unsupervised tasks like dimensionality reduction.

### FCN (Long et al., UC Berkeley, CVPR 2015)

Fully Convolutional Networks can output pixel-wise labels, this is also called **"Dense Prediction"**. The main innovation point is: They leverage a skip architecture to add **appearance information** from shallow, fine layers to a **semantic information** from deep, coarse layers. This mainly comes from the insight [1]:

>  Semantic segmentation faces an inherent tension between semantics and location global information resolves what while local information resolves where.

Fully convolutional computation is famously used in Semernet et al.'s OverFeat, a sliding window approach for object detection. A fully convolutional

#### Receptive Field




The foundation is [transpose convolution](../2017/2017-01-07-transpose-convolution.markdown)



## U-Net (Ronneberger, U Freiburg, MICCAI, 2015)

U-Net is a pioneering image segmentation network primarily designed for biomedical image segmentation (tumors, tissues in lungs, etc.).

The foundation of U-Net is the **encoder-decoder network**, and **FCN**[1]. The innovation in U-Net is skip connections that enables better recovery and spatial details.

Finally, the output is `hxwxk`, where `k` is the output class number.

## References

[1] Long, J., Shelhamer, E., and Darrell, T. 2015. Fully Convolutional Networks for Semantic Segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3431–3440.

[2] Ronneberger, O., Fischer, P., and Brox, T. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). 234–241.
