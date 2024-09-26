---
layout: post
title: Deep Learning - Object Detection Notes
date: '2022-02-09 13:19'
subtitle: Convolution Implementation Of Sliding Window, OverFeat, YOLO V1
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

## Objection

In **Object Detection**, we should be able to get multiple instances of different types of objects. A classic method is the **sliding window method**. With different window sizes, we slide a window across the image. In each window, we run an image classification algorithm to detect if there is any labelled object that could be detected. In the past, image classification was done with hand-engineered feature detection, which is relatively cheap. Nowadays, we are using Conv nets, but they are slower.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/67adfade-3f8c-491c-8953-ba66cca7e1fe" height="200" alt=""/>
    </figure>
</p>
</div>

### OverFeat

In OverFeat [1], a conv net is used for image classification. Here, in the case of `14x14x3` images, the output is `1x1x4` (4 output classes). In the case of `16x16x3`, the output is `2x2x4`. The output gives the location the bounding box at all 4 possible locations in the images, **in just 1 pass**. Before OverFeat, there is only 1 output from the sliding window.

![Screenshot from 2024-09-25 17-22-51](https://github.com/user-attachments/assets/1d4ca366-5ea4-46fd-8b45-32b940c66599)

Interesting technique introduced by OverFeat is that the fully connected layers in the image classification model is replaced by conv layers. The first one is 4 `14x14x3` kernels so the output is `nxnx400` (which is equivalent to 400 neurons in a FC layer). The second one is 1x1 convolution. If we use an FC layer instead, then the output is fixed size (similar to prior work). 

![Screenshot from 2024-09-25 17-06-41](https://github.com/user-attachments/assets/69abd2f1-749c-4134-a6c3-fa4acdb44efb)

In a larger image (`28x28x3`), we can see that due to `max_pooling_window=2, stride=2`, this architecture gives an output of `8x8x4`. That corresponds to all possible locations on the image with a sliding window with a `stride=2`.
![Screenshot from 2024-09-25 17-29-59](https://github.com/user-attachments/assets/8ab14392-f1ec-47df-acc3-4e270d088e77)

OverFeat, however, still suffers from some inaccuracies in bounding box location.

## You Only Look Once (YOLO) V1

The main idea is to divide an image into a `7x7` grid. Each grid will detect the existence of 2 bounding box whose center is within the grid cell with `[p_1, bx_1, by_1, bw_1, bh_1, p_2, bx_2, by_2, bw_2, bh_2, c1, ... c20]`, a `7x7x30` tensor.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/482fc3dd-310c-4e0c-8ded-97f11c735f1f" height="300" alt=""/>
    </figure>
</p>
</div>

The architecture starts off with conv layers, and ends with 2 fully connected (FC) layers. In total, 24 Conv Layers. **The 1x1 convolutions reduce the feature space from preceding layers**. This is very interesting. The first FC layer is connected to the flattened output of the last Conv layer. The last FC layer is reshaped into `7x7x30`.

During model training, the first 20 layers were first trained with on the ImageNet dataset. They were appended with an avg pool layer and a FC layer. This process took **Redmon et al. approx. a WEEK.** Then, they learned from Ren et al. That adding both Conv and FC layers can improve performance. So they added 4 Conv layers with 2 FC layers. Those have randomly initialized weights.

**Loss calculation:** In training, **when calculating loss**, one channel (vector of `30`) is broken into two bounding boxes. Then both bounding boxes are compared against the groundtruth bounding box(es). The ones with the highest Intersection Over Union (IoU) are "responsible" for the corresonding groundtruth bounding box(es). Then, loss can be calculated by adding the weighted confidence loss and localization loss:

Then, because this loss is second order, gradient descent will be first order.

- Localization error if a ground truth bounding box appear in a cell $i$. The responsible predicted bounding box is $j$.  $\lambda_{\text{coord}}=5$ and has a larger weight.

$$
\begin{gather*}
L_{\text{loc}} = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]
\end{gather*}
$$

- Confidence loss: when a ground truth bounding box exists in cell $j$, this penalizes confidence deviations across classes:

$$
\begin{gather*}
L_{\text{conf\_obj}} = \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2
\end{gather*}
$$

- For grids without an object, $\lambda_{\text{noobj}}=0.5$:

$$
\begin{gather*}
L_{\text{conf\_noobj}} = \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2
\end{gather*}
$$

## References
[1] Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann LeCun. 2014. OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks. In Proceedings of the International Conference on Learning Representations (ICLR), 2014.
[2] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016 (pp. 779-788). IEEE.
