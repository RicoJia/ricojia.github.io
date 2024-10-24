---
layout: post
title: Deep Learning - Hands-On UNet Image Segmentation From Scratch
date: '2022-02-19 13:19'
subtitle: UNet
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

[This article is inspired by this referece](https://poutyne.org/examples/semantic_segmentation.html)

## Data Loading

When making a dataset, do NOT use `jpeg` and stick to png instead. `Jpeg` will compress the data and will corrupt the labels.

- Pascal VOC 20 classes, including humans.
- Subtract means and std deviation of ImageNet in pascal is okay since ImageNet is a large enough dataset.
- Since image segmentation target are `uint8`, we should add transform `v2.PILToTensor()` to retain the data type (uint8)
- `v2.Lambda(lambda x: replace_tensor_val(x.long(), 255, 21)),` we are replacing values **255, which is usually ignore index**, to 21. This is because 255 might cause issues in loss function (say softmax) when you have only 20 classes.
  - Alternatively, `loss_function = torch.nn.CrossEntropyLoss(ignore_index=255)` can be used.
- Using `interpolation=InterpolationMode.NEAREST` is important in downsizing, why?
  - Because when downsizing, we need to combine multiple pixels together. Interpolation handles that.
  - Other interpolation methods like bicubic will create continuous float values.

## Training

THE BIGGEST PROBLEM I ENCOUNTERED was the output labels were mostly zero. This is because the dataset is imbalanced and has way many more zeros than other classes. In that case, do not use cross entropy.

- Pascal VOC 2007 has only 209 images for training, 213 images for validation. We thought this would be far from being enough for training. Pascal VOC 2012 has 1,464 images for training, 1449 for validation. However, trying Pascal VOC 2012 did not solve the problem

- `torchvision.transforms.CenterCrop(size)` was necessary because after convolutions, the skip connections are slightly larger than their upsampled peers.

- `focal loss` seems useful, but it was a bit tricky to check. I checked with the one-hot version of output labels, and compare against itself. That was supposed to be the "perfect" example, and I expected to see a loss of 0. However, due to the softmax operation in focal loss, I got 0.78. This was resolved by doing `100 * one_hot_labels`. **Just using focal loss alone did NOT get around the imbalance issue for Pascal VOC 207**

- However, UNet works on the Carvana dataset pretty well, with 97% accuracy (training set only) for a PoC. I haven't measured dev / test set yet.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/3e426cdc-5133-4049-afc0-76044af69b05" height="200" alt=""/>
    </figure>
</p>
</div>

## References

[PyTorch-Unet](https://github.com/milesial/Pytorch-UNet)
