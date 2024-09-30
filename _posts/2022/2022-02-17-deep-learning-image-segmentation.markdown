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

### Fully Convolutional Networks, FCN (Long et al., UC Berkeley, CVPR 2014)

Fully Convolutional Networks (FCN) can output pixel-wise labels, this is also called **"Dense Prediction"**. The main innovation point is: They leverage a skip architecture to add **appearance information** from shallow, fine layers to a **semantic information** from deep, coarse layers. This mainly comes from the insight [1]:

> Semantic segmentation faces an inherent tension between semantics and location global information resolves what while local information resolves where.

To do pixel-perfect labelling:
1. Use shallow conv layers for feature extraction. In these layers, the height and width of feature maps go down (downsampling)
2. Output with dense layer cannot generate pixel-perfect labelling. Therefore, FCN upsamples from learned feature using [**transposed convolution.**](../2017/2017-01-07-transpose-convolution.markdown) The end of the next work is a Transposed Conv
3. Add skip connection betweeen the shallow conv layers and the later deep upsampling conv layers. The shallow layers have spatial information (where features are), but in conventional Conv-Dense hybrid architectures, that information is lost at the dense layers.
4. An **added bonus is the input size is not longer needed to be fixed** like architectures with dense layers.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/869ee426-9d5b-406e-be5e-77a4a8422a4b" height="300" alt=""/>
    </figure>
</p>
</div>

Choices of CNN can be VGG-16, AlexNet, etc. There are three types of outputs:

- FCN-32: the combined downsampling factor is 32. e.g., for an input `512x512`, the feature map that gets downsampled to is `16x16`. The feature map is then directly upsampled back to the original image size. This is very coarse, so FCN-16, FCN-8 are introduced
- FCN-16: The `16x16` feature map first gets upsampled to `32x32`, then it's added with a `32x32` coarse feature map with fine spatial information (from the Pool4 layer). Finally, it gets upsampled back to the original file size by a factor of 16.
- FCN 8: similar to FCN-16, but it finally gets upsampled by a factor of 8.

When combining shallow and deep layer outputs, FCN uses element-wise addition. It's simpler, memory over head is lower. 

Results

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/c8581a89-b4f5-4c39-b8e0-b14beb772fdb" height="200" alt=""/>
    </figure>
</p>
</div>

**Deficiencies of FCN:**

- Information loss at downsampling, so upsampling has trouble learning from it.

Fully convolutional computation is famously used in Semernet et al.'s OverFeat, a sliding window approach for object detection. A fully convolutional Network however is proposed by Long et al.

## U-Net (Ronneberger, U Freiburg, MICCAI, 2015)

U-Net is a pioneering image segmentation network primarily designed for tumor image segmentation.

The foundation of U-Net is the **encoder-decoder network**, and **FCN**[1]. The innovations in U-Net include:
1. Adding a skip connection between every matching downsampling and upsampling block. This allows U-Net to transfer low and high level information more comprehensively.  
2. Using a matching number of convolutions (for downsampling to feature maps) and transposed convolutions (for upsampling back to initial image size). This helps prevent model overfitting.
3. Shallow layers **learns local features** such as edges, corners. Their outputs are responses to the local features with high spatial fidelity (localization), but they lack a global understanding of the scene.
4. Deep layers, after multiple conv+pooling, have larger receptive fields. They capture global context (classes of objects that present in the image) better, but they need localization.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/1c3e2eab-5789-4ffb-a43f-86f0a440397d" height="300" alt=""/>
    </figure>
</p>
</div>

When combining shallow and deep layer outputs, U-Net **concatenates them together**. This preserves full information from both the encoder and the decoder.

### U-Net Architecture Summary

- Contracting Path (decoder)
- Cropping Path
- Expansion Path (encoder)

Finally, the output is `hxwxk`, where `k` is the output class number.

## References

[1] Long, J., Shelhamer, E., and Darrell, T. 2015. Fully Convolutional Networks for Semantic Segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3431–3440.

[2] Ronneberger, O., Fischer, P., and Brox, T. 2015. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI). 234–241.
