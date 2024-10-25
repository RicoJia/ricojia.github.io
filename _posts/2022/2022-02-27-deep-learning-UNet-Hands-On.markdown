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

## Performance Profiling

- For the GTA5 dataset, my `train/dev/test` dataset split is `70%, 15%, 15%`. My accuracies are
  - train: `68.1%`
  - dev: `68.7%`
  - test: `67.88%`
    - examples

        <div style="text-align: center;">
        <p align="center">
            <figure>
                <img src="https://github.com/user-attachments/assets/5ddb1179-434a-4822-9d80-056df798f868" height="400" alt=""/>
            </figure>
        </p>
        </div>

        <div style="text-align: center;">
        <p align="center">
            <figure>
                <img src="https://github.com/user-attachments/assets/e7fa4efa-f5a7-4c9b-a8ed-2c769bdd30b3" height="400" alt=""/>
            </figure>
        </p>
        </div>


        <div style="text-align: center;">
        <p align="center">
            <figure>
                <img src="https://github.com/user-attachments/assets/c86ab2a9-65f8-4463-b12f-27b6164e43d1" height="400" alt=""/>
            </figure>
        </p>
        </div>

- Cavana dataset: my `train/dev/test` dataset split is `70%, 15%, 15%`. My accuracies are:
  - Mixed precision training (average 383s/batch)
    - train: `90.51%`
    - dev: `90.46%`
    - test: `90.61%`
  - FP32 Full precision training (time)
    - train: `90.55%`
    - dev: `90.63%`
    - test: `90.66%`
- Pascal VOC 2007
    - Mixed precision training (average 383s/batch)
        - train: `72.97%`
        - dev: `73.61%`
        - test: `74.27%`
    - Examples:

        <div style="text-align: center;">
        <p align="center">
            <figure>
                <img src="https://github.com/user-attachments/assets/c3353c3f-308c-4b81-9798-8873b2488b39" height="200" alt=""/>
            </figure>
        </p>
        </div>

        <div style="text-align: center;">
        <p align="center">
            <figure>
                <img src="https://github.com/user-attachments/assets/2f2538f8-0822-4da8-b8b3-b5460531b20d" height="200" alt=""/>
            </figure>
        </p>
        </div>


        <div style="text-align: center;">
        <p align="center">
            <figure>
                <img src="https://github.com/user-attachments/assets/00e629bb-d286-4d28-bea1-8e74c553eb36" height="200" alt=""/>
            </figure>
        </p>
        </div>

## References

[PyTorch-Unet](https://github.com/milesial/Pytorch-UNet)
