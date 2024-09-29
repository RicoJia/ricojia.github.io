---
layout: post
title: Deep Learning - Object Detection Notes
date: '2022-02-13 13:19'
subtitle: YOLO V1, R-CNN
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## You Only Look Once (YOLO) V1

The main idea is to divide an image into a `7x7` grid. Each grid will detect the existence of 2 bounding box whose center is within the grid cell and outputs `[p_1, bx_1, by_1, bw_1, bh_1, p_2, bx_2, by_2, bw_2, bh_2, c1, ... c20]`, a `7x7x30` tensor. bounding box size parameters, `bw, bh` are percentages relative to the corresponding image width and height. 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/482fc3dd-310c-4e0c-8ded-97f11c735f1f" height="300" alt=""/>
    </figure>
</p>
</div>

The architecture starts off with conv layers, and ends with 2 fully connected (FC) layers. In total, 24 Conv Layers. **The 1x1 convolutions reduce the feature space from preceding layers**. This is very interesting. The first FC layer is connected to the flattened output of the last Conv layer. The last FC layer is reshaped into `7x7x30`.

During model training, the first 20 layers were first trained with on the ImageNet dataset. They were appended with an avg pool layer and a FC layer. This process took **Redmon et al. approx. a WEEK.** Then, they learned from Ren et al. That adding both Conv and FC layers can improve performance. So they added 4 Conv layers with 2 FC layers. Those have randomly initialized weights.

**Loss calculation:** In training, **when calculating loss**, one channel (vector of `30`) is broken into two bounding boxes. Then both bounding boxes are compared against the groundtruth bounding box(es). The ones with the highest [Intersection Over Union (IoU)](../2021/2021-01-05-computer-vision-non-maximum-suppression.markdown) are "responsible" for the corresonding groundtruth bounding box(es). Then, loss can be calculated by adding the weighted confidence loss and localization loss:

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

### Anchor Boxes

In YOLO V2, V3, and V4, Aa anchor box is a pre-defined bounding box that "anchors" to a cell. It has a pre-defined aspect ratio and a size. For example, if we define a 3x3 grid over an image, at each grid cell, we can define 3 anchor boxes: 1:1 small square, 2:1 tall rectangle, and 1:2 wide rectangle. 

During training, the goal is to leared the best `[b_w, b_h]` where `b_w, b_h` are percentages relative to the **pre-defined anchor box sizes**

During inferencing, if a grid cell has an object in it, the model will output `[confidence, location offset, aspect ratio offset]` to closely fit the object.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e8f06970-164d-4118-a623-f4d280e0f097" height="200" alt=""/>
    </figure>
</p>
</div>

In YOLO V2, each object is assigned to the grid cell that has the object's midpoint. The object is also assigned to the anchorbox with the highest IoU (between the groudtruth and the detected boxes) inside the cell. It's also used in Faster R-CNN and SSD. 

### Comparison Between AlexNet, OverFeat, and YOLO V1

AlexNet: `Conv+max|Conv+max|Conv|Conv+max|Dense|Dense|Dense|`

OverFeat: `Conv+max|Conv+max|Conv|Conv+max|Dense|Dense|Dense|`

YOLO V1: `Conv+max|Conv+max|bottleneck Conv + max block | bottleneck Conv+max block | bottleneck Conv + max block | Conv Block | Dense | Dense |`

So OverFeat and AlexNet’s architectures looks very similar. AlexNet is just image classification, while Overfeat is image classification + object detection.

YOLO V1 (24 conv layers + 2 FC layers) is larger than AlexNet and OverFeat. 1 forward pass in Overfeat is equivalent to sliding one fixed-size window. YOLO V1 however, can output window of any size (the network will learn the window size from the training data)

## Region Based CNN (R-CNN, Girshick et al. CVPR 2014)

[Zhihu](https://zhuanlan.zhihu.com/p/383167028)

Regional Proposal is the core of R-CNN. It first uses a segmentation algorithm to find regions with objects, then use these regions as "region proposals" for CNN to run on [2]. Each region outputs `[label, bounding box]`. 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e48cefc4-d7b9-4bb2-9a03-5a8ecebeff45" height="200" alt=""/>
    </figure>
</p>
</div>

1. Use Selective Search Algorithm to come up with 2000 region proposals: TODO
    - Use Hierarchical Grouping Algorithm  (Felzenszwalb and Huttenlocher, 1999)
        TODO: https://zhuanlan.zhihu.com/p/39927488
2. Use AlexNet for Feature Extraction on 2000 region proposals.
    - Get 2000x4096 feature vector
3. Classification & bounding box
    - Use 21 SVM to classify 21 classes (including background) on 2000 region proposals
        - Each SVM has 21 values.
    - Parallel to classication, use TODO regression for bounding box regression

Later came Fast R-CNN (Girshick, ICCV 2015). Fast R-CNN propose regions, then uses convolution implementation of sliding windows to classify all proposed regions.

Then came Faster R-CNN (Ren, He et al. NeurlPS 2015). They are all slower than YOLO. From Prof. Andrew Ng's perspective, YOLO's 1-stage architecture is more concise.

- YOLOv3: 30 fps on high end CPU
- Faster R-CNN: 7 fps+
- YOLOv4 and YOLOv5: 60fps+

## References
- [1] [Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016 (pp. 779-788). IEEE.](https://arxiv.org/pdf/1506.02640)
- [2] [R. Girshick, J. Donahue, T. Darrell, and J. Malik, "Region-Based Convolutional Networks for Accurate Object Detection and Segmentation," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 580–587.](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
