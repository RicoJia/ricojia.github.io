---
layout: post
title: Deep Learning - Classic CNN Models
date: '2022-02-03 13:19'
subtitle: LeNet-5, AlexNet, VGG-16, ResNet
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## LeNet-5 (65K params)

The LeNet-5 architecture (LeCun, 1998) and is still very widely used. 5 is simply the version number. LeNet-5 has 7 layers.

1. Input layer are 32x32 grayscale images. MNIST images are originally 28x28, but here they are padded to 32x32.
2. The next 4 layers are CNN layers without padding. Each CNN is attached to an Average Pooling layer. Nowadays, we usually use Max Pooling Layers
3. The 6th layer is a FCN that connects to each output pixel of the last CNN. Activation is Sigmoid.
4. The 7th layer is another FCN layer with 10 neurons that uses a non-popular classifier (Euclidean Radial Basis Function, RBF). Nowadays, we use Softmax.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/57b7026e-701e-49a3-9505-5f9f261ba6bb" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>

Other architectural aspects:
- Activation functions were **Sigmoid and Tanh**, rather than ReLU (ReLU came out with AlexNet).
- Nowadays, a CNN layer would have `f x f x c` for each output channel, where `f` is the filter size, `c` is the number of input channels. But LeNet-5 creates filters of different sizes for different channels for the **small compute it had**.
- LeNet-5 attaches a non-linear activation function after a CNN.
- Nowadays, we often use max pooling instead of average pooling.

## AlexNet (65M Params)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/208d3b4b-dcaa-4446-b8fe-bb9ca25ac896" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>

Input: `227x227x3`. From the image net dataset

- 1st layer: CNN with 11x11, stride = 4. Output: `55x55x96`
- 2nd layer: Maxpooling layer with 3x3 kernel size, stride=2. Output `27x27x96`
- 3nd layer: CNN with Same Cross-Correlation. Output: `27x27x256`
- 4th layer: Maxpooling Layer with kernel size 3x3, stride = 2
- 5th, 6th, 7th are CNN layers with Same Cross-Correlation
- 8th layer is a maxpooling layer with a kernel size 3x3, stride = 2
- 9th layer is a flattening layer
- 10th and 11th layer is A FCN. Output are `4096 x 1`
- 12th layer is a FCN layer with 1000 output classes. Its outputs are then fed into a softmax

Highlights include:
- Trained on a **much larger network** than LeNet-5
- **Use of ReLU**

Historical Constraints:

- Back then GPU was slower, and the training of layers were split on 2 GPUs.
- Local Response Normalization: normalize the **same patch of a layer across channels** to avoid high activation values. But Later people didn't find it universally helpful

AlexNet paper is relatively easy to read. And it's a milestone that drew people's serious attention in neuralnetwork.

## VGG-16 (138M parameters)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/72119760-f953-4e05-9f80-fc150fa094e9" height="300" alt=""/>
        <figcaption><a href="https://lekhuyen.medium.com/an-overview-of-vgg16-and-nin-models-96e4bf398484">Source: Le Khuyen</a></figcaption>
    </figure>
</p>
</div>

There are **16 layers that have parameters**. The architecture of VGG 16 is realitively simpler.

Upside is the architecture is **quite simple**. Downside is it's large even by modern standard 

## ResNet

Deep Network suffer from exploding and vanishing gradients. ResNet created the notion of Residual Blocks, which is:

```python
a_l -> z_l+1 = w^T a_l + b-> a_{l+1} = g(z_l+1) -> z_{l+2} = w^T a_(l+1) -> a_{l+2} = g(z_l+1 + a_{l})
```

The difference is that we add `a_{l}` to the non linear activation function of `a_{l+2}`. 
This residual block basically "boosts" the original signal. Such connections are also called "skip connections or short circuits".

In reality, deep plain networks will witness an increase in training errors without resnet. But ResNet overcomes this problem.

TODO: plot in resnet training error going back up

This is because when weights and biases are zero, we can still get the signals from the previous layers, so our behavior is at least as good as the previous ones.

## One by One Filters
1x1xn filter

It can shrink (summarize) the number of channels. So it can yield from nxnxm to nxnxc


## Inception Network

Inception network can do a combo of conv and pooling, or one of these. You can append outputs of different output channels together. E.g., one can append 64 output layers from 1x1 conv, and 128 layers from 3x3 (same conv) together.

So, if the number of conv filters in use is learnable.
Question: How does padding in max pooling work? Appending zeros

![Screenshot from 2024-09-19 22-06-01](https://github.com/user-attachments/assets/760d0d88-56cc-421e-aac7-959c27c83509)


### Problem With Covolution Layer
Convolution is expensive. Below network is (5x5x28x28)x192x32 multiplication in its forward prop ( prod of dimensions of input and filters). 1x1 conv can effectively reduce the number of multiplications. The associated layer is called “bottleneck layer”

![Screenshot from 2024-09-19 22-06-28](https://github.com/user-attachments/assets/ca0d77b2-afb6-4041-b2e8-0ddad0382f60)

"Bottleneck Layer"? TODO

![Screenshot from 2024-09-19 22-06-57](https://github.com/user-attachments/assets/0546d9e8-7f59-47e3-ae69-67944c273495)

Max pool layer itself will output the same num of channels. So we need 1x1 conv to shrink down

