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

Data

- Pascal VOC 20 classes, including humans.
- Subtract means and std deviation of ImageNet in pascal is okay since ImageNet is a large enough dataset.
- Since image segmentation target are `uint8`, we should add transform `v2.PILToTensor()` to retain the data type (uint8)
- `v2.Lambda(lambda x: replace_tensor_val(x.long(), 255, 21)),` we are replacing values **255, which is usually ignore index**, to 21. This is because 255 might cause issues in loss function (say softmax) when you have only 20 classes.
    - Alternatively, `loss_function = torch.nn.CrossEntropyLoss(ignore_index=255)` can be used.
- Using `interpolation=InterpolationMode.NEAREST` is important in downsizing, why?
    - Because when downsizing, we need to combine multiple pixels together. Interpolation handles that.
    - Other interpolation methods like bicubic will create continuous float values.