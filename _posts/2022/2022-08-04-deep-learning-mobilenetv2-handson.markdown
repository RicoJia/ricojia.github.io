---
layout: post
title: Deep Learning - MobilenetV2 Hands-On
date: '2022-08-01 13:19'
subtitle: Inverted Skip Connection, Multiclass Classification
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
    - Hands-On
---

## MobileNet Architectures

In MobileNet V1, the architecture is quite simple: it's first a conv layer, then followed by 13 sets of depthwise-separable layers. Finally, it has an avg pool, FC, and a softmax layer. In total there are 1000 classes.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/4dc56f26-3abe-42e8-a29e-2a636407e40a" height="200" alt=""/>
    </figure>
</p>
</div>

In MobileNet V2 (Sandler et al., 2018), the biggest difference is the introduction of the "inverted bottleneck block". The interverted bottleneck block adds a skip connection at the beginning of the next bottleneck block. Additionally, an expansion and projection are added in the bottleneck block. MobileNet V2 has 155 layers.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8eff937b-241e-4a69-b877-a1ed564efa7f" height="150" alt=""/>
    </figure>
</p>
</div>

In a bottleneck block,

1. dimensions are jacked up so the network can learn a richer function.
2. perform depthwise convolutions
3. they are projected down before moving the next bottleneck block so memory usage won't be too big, and model size can be smaller

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/32afa29b-3146-4dd8-9dee-de772b633f70" height="200" alt=""/>
    </figure>
</p>
</div>

Overall, MobileNet V2 has 155 layers (3.5M params). There are 16 inverted residual (bottlenck) blocks. Some blocks have skip connections to the block after the next one. Usually, in each bottleneck block, after an expansion and a depthwise convolution, there is a batch normalization and ReLu. After a downsizing projection, there is a batch normalization.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e9e5502a-8c50-4b71-8791-a8a1c35e88b2" height="400" alt=""/>
    </figure>
</p>
</div>

## Data

- We are using the **COCO dataset** to train a multi-class classifier with a MobileNetV2. [The dataloading code can be found here](https://github.com/RicoJia/Machine_Learning/blob/b309416af67339a67612a8f99ea692117b9ebca6/RicoModels_pkg/ricomodels/utils/data_loading.py#L380)

- A common way to frame this problem is to convert it into a binary classification problem. The way to do is to create a multi-hot vector where a class is 1.0 if it's in the class list

```
(class_1_probability, class_2_probability ...)
[1.0, 0.5, 1.0, ...]
```

During training, we will calculate Binary Cross Entropy loss on this multi-hot vector. See the Model section for more details.

## Model

- In this implementation, we added dilated convolution as well to increase the receptive field range.

- Initialization method is conventional. We are doing:
  - `Conv 2D` layers:
    - `He` initialization on weight matrices
    - `Zero` initialization on biases
  - `Linear` layers:
    - `Normal` initialization with `mean=0, std_dev=1.0`
    - `Zero` initialization on biases
  - `Batch norm` layers:
    - `Zero` initialization on mean,
    - `One` initialization on `std_dev`

- `ReLu6` is more robust in low-precision training?
- It's generally a good idea to use `in_place=True` to avoid `out of memory` error.
- [MobileNet V2 uses dilated convolution to increase receptive fields](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/4e1087de98bc49d55b9239ae92810ef7368660db/network/backbone/mobilenetv2.py#L68)
- `torch.nn.Conv2d(groups)`:
  - At groups=1, all inputs are convolved to all outputs.
  - At groups=2, the operation becomes equivalent to having two conv layers side by side, each seeing half the input channels and producing half the output channels, and both subsequently concatenated.
  - At groups= in_channels, each input channel is convolved with its own set of filters.
- `FP16 & float32` mixed precesion training was also used.

- At the end of the model, we have one dense layer as a "classifier".
  - The classifier does NOT have sigmoid at the end. This is trained for multi-label classification. There should be a sigmoid (not softmax) layer following it, but it is handled in the nn.BCEWithLogitsLoss.
  - The classifier dim is the same as number of classes in the dataset. So if we train across multiple datasets, we have to map the datasets correctly.
- Model Summary: 141 layers and 2.3M parameters. This is a modified version from the original model.

```
Total params: 2,326,352
Trainable params: 2,326,352
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 1890.44
Params size (MB): 8.87
Estimated Total Size (MB): 1900.06
```

## Training Adjustments & Iterations

### [1] Training with the `nn.BCELossWithLogits()` loss

When considering accuracy on both positives and negatives, accuracies are: `training set 97.97%`, `validation set 97.73%`. **But I realized that we have way too many negatives**. The f1 score were: `training set 0.654`, `validation set 0.617`.

- Observations: I don't see overfitting or underfitting.
- Actions
  - Check for better loss options
    - F1 score [Not suitable because it's not differentiable](./2022-01-08-deep-learning-Activation-Losses.markdown)
        - [Could try F1 Metric Variant](https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/comments)
    - Focal loss (✅)
  - Check for training images. **Some images are dark** after normalization with ImageNet's mean and std_dev.
  - Fine tuning:
    - freeze early layers. early layers capture local features. Freezing the late layers might be beneficial
