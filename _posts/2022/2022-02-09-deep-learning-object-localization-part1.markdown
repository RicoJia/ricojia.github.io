---
layout: post
title: Deep Learning - Object Detection Notes Part 1
date: '2022-02-09 13:19'
subtitle: Convolution Implementation of Sliding Window, OverFeat
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction

Image classification is given an image, output a class lable of the image
Image classification and Object localization is Object Detection.

For example, if there are 4 classes, `[Pedestrian, Cars, Motorcycles, Background]`, we expect an output in `[p, cx, cy, cz, bx, by, bw, bh]`.

- p is the **confidence of** whether there is an object (so background would make this False). `cx, cy, cz` is the one-hot vector of the output class label. `bx, by, bw, bh` are the (x,y) and (width, height) of the detected object.

**Loss function** can be $L(\hat{y},y) = (\hat{p}-p)^2$ if the label value for $p$ is 0. If the label value for $p$ is 1, then the loss function can be the summed squared error $L(\hat{y},y) = (\hat{p}-p)^2 + (\hat{c_x}-c_x)^2 + ...$. This is almost **a regression task**

**Landmark** is a salient feature of an image we want to recognize. E.g., certain corners of the eye, edges along a face, etc. This is important for Snapchat's filters, such as the one that adds a crown on a person's head. So the labelled data is a list of **consistent indexed** landmarks, for example, "landmark 1 is always the left corner of the left eye" (generating that could be a laborious process). The output of the neuralnet is `[x_1, y_1, ... x_64, y_64]` if we want 64 landmarks. Landmark Detection Will pave the way for object detection

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/1bef1087-5ffb-40a4-a87e-b749913a526b" height="150" alt=""/>
    </figure>
</p>
</div>

## Object Detection

In **Object Detection**, we should be able to get multiple instances of different types of objects. A classic method is the **sliding window method**. With different window sizes, we slide a window across the image. In each window, we run an image classification algorithm to detect if there is any labelled object that could be detected. In the past, image classification was done with hand-engineered feature detection, which is relatively cheap. Nowadays, we are using Conv nets, but they are slower.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/67adfade-3f8c-491c-8953-ba66cca7e1fe" height="200" alt=""/>
    </figure>
</p>
</div>

### OverFeat (Sermanet et al., 2013)

In OverFeat [1], a conv net is used for image classification. Here, in the case of `14x14x3` images, the output is `1x1x4` (4 output classes). In the case of `16x16x3`, the output is `2x2x4`. The output gives the location the bounding box at all 4 possible locations in the images, **in just 1 pass**. Before OverFeat, there is only 1 output from the sliding window.

![Screenshot from 2024-09-25 17-22-51](https://github.com/user-attachments/assets/1d4ca366-5ea4-46fd-8b45-32b940c66599)

Interesting technique introduced by OverFeat is that the fully connected layers in the image classification model is replaced by conv layers. The first one is 4 `14x14x3` kernels so the output is `nxnx400` (which is equivalent to 400 neurons in a FC layer). The second one is 1x1 convolution. If we use an FC layer instead, then the output is fixed size (similar to prior work).

![Screenshot from 2024-09-25 17-06-41](https://github.com/user-attachments/assets/69abd2f1-749c-4134-a6c3-fa4acdb44efb)

In a larger image (`28x28x3`), we can see that due to `max_pooling_window=2, stride=2`, this architecture gives an output of `8x8x4`. That corresponds to all possible locations on the image with a sliding window with a `stride=2`.
![Screenshot from 2024-09-25 17-29-59](https://github.com/user-attachments/assets/8ab14392-f1ec-47df-acc3-4e270d088e77)

OverFeat, however, still suffers from some inaccuracies in bounding box location.

#### OverFeat Classification and Localization

I found some slides [from Stanford CS231](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/overfeat_eric.pdf). They say “the fully connected layer of the classifier is replaced by a regressor”. Then freeze the classifer, and train the network again on labeled input with bounding boxes.

- The regressor will finally output [(x,y) of top left, top right corners] .  The regressor network is as follows:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://global.discourse-cdn.com/dlai/original/3X/4/3/43219e22c2ccfb4f44e6c50f0f7546e883f17906.png" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>

After a quick scan, I didn’t see the how training is done in the first author, [Sermanet’s C++ implementation though](https://github.dev/sermanet/OverFeat/tree/master/src)

So to figure out what the regressor network really look like, I found this [YouTube Video](https://www.youtube.com/watch?app=desktop&v=JKTzkcaWfuk) that came up with an explanation that looks mostly reasonable to me, but **please take it with a grain of salt**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://global.discourse-cdn.com/dlai/original/3X/2/c/2c2240374e33ab56c77d6d555b1e34e31034626e.png" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>

**I’m not sure about the 1x1x4096 implementation though**

My understanding of the regressor network is:

    First train the classification network. Freeze it and add the localizer network in.
    Layer 1 in the regressor has input 6x7x256, output 2x3x4096, so I believe it’s 5x5x256x4096 (Convolutional)
    Layer 2: input: 2x3x4096, output 2x3x1024. This looks like an 1x1x1024 convolution to me?
    Output layer: input: 2x3x1024, output 2x3x4. So this can be also achieved by 1x1x4 convolution?

I put question marks at the 1x1 convolution is 1x1 conv was introduced by Network In [Network (NiN) Architecture (Lin et al., 2013)](http://d2l.ai/chapter_convolutional-modern/nin.html), and was heavily used in GooLeNet in 2014. However I don’t see either of these in the OverFeat paper’s reference.

## References

- [1] [Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann LeCun. 2014. OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks. In Proceedings of the International Conference on Learning Representations (ICLR), 2014.](https://arxiv.org/pdf/1312.6229)
