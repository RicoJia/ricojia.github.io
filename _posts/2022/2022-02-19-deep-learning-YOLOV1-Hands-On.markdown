---
layout: post
title: Deep Learning - Hands-On YOLO V1 Transfer Learning
date: '2022-02-19 13:19'
subtitle: YOLO V1 Theory & Transfer Learning
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

## YOLO V1 Implementation

### Architecture

```
Image (608, 608, 3) -> Deep CNN -> Encoding (m, n_h=19, n_w=19, anchors=5, classes=80) -> non max suppresion -> [p, bx, by, bh, bw, C]
```

- The 2nd to last layer is the encoding layer: `[p, bx, by, bh, bw, c_1, c_2 ... ]` of 5 anchor boxes for each grid cell. **For simplicity, we flatten the anchors into a 425 vector**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/f05782e0-23db-459f-9506-3cf7142f5cc0" height="300" alt=""/>
        <figcaption> Encoding Represents 5 boxes at each grid cell </figcaption>
    </figure>
</p>
</div>

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/b39bd894-21fb-4a9d-8dc9-f05c9564a245" height="300" alt=""/>
        <figcaption> Dimension Flattening </figcaption>
    </figure>
</p>
</div>

- The probability of a cell containing an object of a given class is $p(class, object) = p(class | object) * p(object)$.
- Anchor boxes are pre-assigned.
- Can use labels or a single integer. In the above illustration, we are using a single integer.
- `Image -> CNN -> (19, 19,425)`, where `425 = 5 * 85. 85 = 5($(p_c, b_x, b_y, b_h, b_w)$) + 80 (classes)`

### Post Processing

yad2k is a custom library developed by [Allan Zelener](https://github.com/allanzelener/YAD2K/tree/master)

```python
# in yad2k/models/keras_yolo.py
yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
```

### Visualization

One way to visualize the output of each grid cell is to plot the corresponding classes in colors

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/5bd1fa27-185f-4dbc-a3de-80c24ce60418" height="300" alt=""/>
    </figure>
</p>
</div>

- Another way is to visualize the bounding boxes before NMS

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/4b6fe18d-0e4e-4804-b72f-cd2c8fcd7f7d" height="300" alt=""/>
    </figure>
</p>
</div>
