---
layout: post
title: Deep Learning - Classic CNN Models
date: '2022-02-03 13:19'
subtitle: LeNet-5, AlexNet, VGG-16, ResNet, One-by-One Convolution, Inception Network, MobileNet
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
        <img src="https://github.com/user-attachments/assets/57b7026e-701e-49a3-9505-5f9f261ba6bb" height="200" alt=""/>
    </figure>
</p>
</div>

Other architectural aspects:
- Activation functions were **Sigmoid and Tanh**, rather than ReLU (ReLU came out with AlexNet).
- Nowadays, a CNN layer would have `f x f x c` for each output channel, where `f` is the filter size, `c` is the number of input channels. But LeNet-5 creates filters of different sizes for different channels for the **small compute it had**.
- LeNet-5 attaches a non-linear activation function after a CNN.
- Nowadays, we often use max pooling instead of average pooling.

## AlexNet (65M Params, Alex Krizhevsky 2012)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/208d3b4b-dcaa-4446-b8fe-bb9ca25ac896" height="300" alt=""/>
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

## One by One Convolutions

An 1x1 filter is mainly used to shrink (summarize) or expand the number of channels, while the height and width of the feature maps remain the same (`nxnxm` -> `nxnxc`). This technique is also called **"pointwise convolution"**.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/febf43af-a45f-41c8-ba6f-2606b75a3072" height="300" alt=""/>
        <figcaption><a href="https://community.deeplearning.ai/t/difference-between-1-1-convolution-and-pointwise-convolution/149338">Source: Deeplearning.ai</a></figcaption>
    </figure>
</p>
</div>

So if we have `pxqxm` input, we would have an `1x1xc` filter. Each output channel of the filter has a `1x1xm` kernel. Elements across all channels at the same `[x,y]` would sum up and multiply by the value at channel `n` of the filter. That's the output value at channel `n` is determined.

$$
\begin{gather*}
\begin{bmatrix}
1 & 2 & 3 & 6 & 5 & 8 \\
3 & 5 & 5 & 1 & 3 & 4 \\
2 & 1 & 3 & 4 & 9 & 3 \\
4 & 7 & 8 & 5 & 7 & 9 \\
1 & 5 & 3 & 7 & 4 & 8 \\
5 & 4 & 9 & 8 & 3 & 5
\end{bmatrix}
\quad \times \quad
\begin{bmatrix}
2 & 4
\end{bmatrix}
\quad = \quad
\begin{bmatrix}
2 & 4 & 6 & \cdots \\
\vdots & & & \vdots \\
\end{bmatrix}
\quad
\begin{bmatrix}
4 & 8 & 12 \cdots \\
\vdots & & & \vdots \\
\end{bmatrix}
\end{gather*}
$$

Winner of ImageNet Large Scale Visual Recognition Challenge (ILSVRC), GoogleNet(2014), ResNet and SqueezeNet all use one-by-one convolution as a major part of the network.

## ResNet-50 (25M, Kaiming He et al. 2015)

Deeper neuralnets in general are favourable: they are able to learn highly non-linear decision boundaries. When building deep neuralnets, Deep Network suffer from exploding and vanishing gradients. In the work Deep Residual Learning for Image Recognition, Kaiming He et al. introduced ResNet. ResNet-50 has in total 50 learnable layers. Compared to "plain networks", ResNet created the notion of **Residual Blocks** that makes uses of "skip connection" or "short circuits" (ResNet-34):

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/333dc157-0d6b-4fa0-90bd-cfbd65ec81d4" height="300" alt=""/>
    </figure>
</p>
</div>

[Code of PyTorch Implementation in Torch Vision](https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/resnet.py#L75)

$$
\begin{gather*}
a_l -> z_{l+1} = w^T a_l + b-> a_{l+1} = g(z_l+1) -> z_{l+2} = w^T a_{l+1} -> a_{l+2} = g(z_{l+1} + a_{l})
\end{gather*}
$$

The difference is that we add `a_{l}` to the non linear activation function of `a_{l+2}`.

This residual block basically "boosts" the original signal. In reality, deep plain networks suffer from two difficulties:

1. It will witness an increase in training errors without resnet. This might be due to vanishing / exploding gradients
2. Optimization difficulties

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/6635f6d8-a4be-4638-8eca-1220f5411d59" height="150" alt=""/>
        <figcaption><a href="https://medium.com/data-science-community-srm/residual-networks-1890cec76dea">Source: Aditya Mangla </a></figcaption>
    </figure>
</p>
</div>

Another highlight in He et al.'s work is "bottleneck building block" architecture (ResNet-50):

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/7ee91657-c25c-4218-8e01-6d9dcbece339" height="200" alt=""/>
        <figcaption><a href="https://arxiv.org/pdf/1512.03385">Source: Deep Residual Learning for Image Recognition, Kaiming He et al.</a></figcaption>
    </figure>
</p>
</div>

The bottleneck building block is to reduce the number of parameters while increasing the number of layers. The 1x1 conv layers reduce / increase channel dimensions, so the actual 3x3 conv layer has smaller dimensions to work with.

### Why ResNet Works

1. ResNet is able to learn "identity" when it's optimal to do so. That is, a residual block's $w$ and $b$ could be both zeros. So the final result will be no worse than that of a plain network. This requires the **input & output dimensions to match**

- The original paper proposed "projection shortcut" as well, which is used when input & output dimensions do not match $H(x) = F(x) + W_sX$. However, this seems to be performing worse than the identity shortcut $H(x)=F(x)+X$ in the bottleneck building blocks.

2. Each residual block's parameters are smaller, and the learned function is simpler. Given input $x$, in a plain network, a layer will learn an entire transformation $H(x)$. However in a residual block, it will learn $F(x)$ where $H(x) = X+F(x)$. There is a chance that the residual $F(x)$ is close to zero. Hence the parameters are smaller (note, **not fewer**). This is especially true when "identity" is the optimal transform $H(x)$.

3. Gradient flow can reach deeper. With skip connections, the input to a layer $x$ has more influence on the final output with less layers to go though $W_1W_2...x$. So gradients will be correspondingly higher and the vanishing gradient problem is mitigated.

## Inception Network (Szegedy et al. 2014, Google, Going deeper with convolutions)

### Motivation

When hand-picking conv nets, we need to specify filter dimensions by hand. Inception network allows **a combo of conv and pooling layers**. Then, outputs in different output channels (depth) are stacked together. E.g., one can append 64 output layers from 1x1 conv, and 128 layers from 3x3 (convolution with `same` padding) together. The name "inception" was adopted by the GoogleNet team in 2014 to pay homage to the movie "inception" we need to go DEEPER.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/760d0d88-56cc-421e-aac7-959c27c83509" height="300" alt=""/>
        <figcaption>Output Channels From Different Feature Maps Are Stacked Together</figcaption>
    </figure>
</p>
</div>

### How Padding in 'Same' Max Pooling Works

In the Inception Network, we need to retain the same output dimension. The trick is to pad the input first, then apply max pooling with the specified stride. E.g., given an input

$$
\begin{gather*}
\begin{bmatrix}
1 & 3 & 2 & 1 \\
4 & 6 & 5 & 7 \\
2 & 1 & 4 & 9 \\
1 & 3 & 7 & 6
\end{bmatrix}
\end{gather*}
$$

If we apply maxpooling with stride = 1, kernel `2x2`, the padded input would be:

$$
\begin{gather*}
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 3 & 2 & 1 & 0 \\
0 & 4 & 6 & 5 & 7 & 0 \\
0 & 2 & 1 & 4 & 9 & 0 \\
0 & 1 & 3 & 7 & 6 & 0 \\
0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
\end{gather*}
$$

The maxpooling result would be:

$$
\begin{gather*}
1 & 3 & 3 & 7 \\
6 & 6 & 5 & 7 \\
6 & 7 & 9 & 9 \\
3 & 7 & 7 & 9
\end{gather*}
$$

### Problem With Covolution Layer
Convolution is expensive. Below network has `(5x5x28x28)x192x32=120422400` times of multiplication in its forward prop (product of dimensions of input and output). 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/ca0d77b2-afb6-4041-b2e8-0ddad0382f60" height="300" alt=""/>
    </figure>
</p>
</div>

1x1 conv can effectively reduce the number of multiplications: `28x28x192x16 + 28x28x16x5x5x32=12443648` multiplications.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/0546d9e8-7f59-47e3-ae69-67944c273495" height="300" alt=""/>
    </figure>
</p>
</div>
The layer is called “bottleneck layer” because of the shape of the 1x1 layer compared to other larger layers.  
Max pool layer itself will output the same num of channels. So we need 1x1 conv to shrink down

### Inception Network Architecture

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/50c8d6c1-63db-459c-83f3-488607e8f79e" height="300" alt=""/>
    </figure>
</p>
</div>

1x1 Conv is used to shrink (or summarize) the outputs from each conv sub-layer.

Szegedy et al. proposed `GoogLeNet`, an Inception Network. The first few layers are regular convolutional layers, followed by inception modules.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/db212e75-d7a5-4240-9331-a0a89cf965a3" height="500" alt=""/>
    </figure>
</p>
</div>

There are two side branches with two softmax outputs. Each side branch is called an **auxilary classifier**. Instead of training separately, they are trained together with the main output. During back-propagation, the connected layers will be updated with the combined gradient from the main and auxilary branches. **This is to compensate for the vanishing gradient problem, so intermediate layers won't be too bad.**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/3e6b2135-d97d-4888-9f43-56a6bcaa121e" height="300" alt=""/>
        <figcaption><a href="">Source: </a></figcaption>
    </figure>
</p>
</div>


## MobileNet (Howard et al. )

Mobilenet v1: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

### Depthwise Separable Convolution

If you have an input of size a×b×3 and a regular convolution filter of size 3×3×5, you would have 3×3×3 filters for each of the 5 output channels. The output would be a×b×5, after summing across input channels.

In a **depthwise convolution**, you would have 3 filters of size 3×3 (one for each input channel). Each input channel is **independently** convolved with its filter, so the output would remain a×b×3a×b×3 (same number of channels as the input).

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/1ba45f51-ba96-435e-9b7d-ad82ed596f82" height="200" alt=""/>
    </figure>
</p>
</div>

Point-wise convolution is just another term for one-by-one multiplication

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/a4b57a73-740f-46fd-ae2c-fd39666dca34" height="200" alt=""/>
    </figure>
</p>
</div>

**The combination of Depthwise convolution and pointwise convolution is called "depthwise separable convolution"**. That reduces the number of multiplications by quite a bit. E.g., when our input is `6x6x3`, we want our output is  `4x4x5`

- In normal convolution, this would take 5 `3x3x3` filters. In total we need `3x3x3x4x4x5=2160` multiplications. 
- Using depthwise separable convolution, we first have 1 `3x3x3` filters, then we have 5 `1x1x3` filters. That in total we have `4x4x3x3x3 + 4x4x3x5=672`. The ratio of the number of multiplications **between mobile net and regular convolution** is:

$$
\begin{gather*}
\frac{1}{f^2} + \frac{1}{m}
\end{gather*}
$$

where $m$ is the number of output channels, $f$ is the filter size. In some applications, $m$ is much larger, so the ratio is slightly larger than $\frac{1}{f^2}$

### MobileNet Architectures

In MobileNet V1, the architecture is quite simple: it's first a conv layer, then followed by 13 sets of depthwise-separable layers. Finally, it has an avg pool, FC, and a softmax layer.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/4dc56f26-3abe-42e8-a29e-2a636407e40a" height="300" alt=""/>
    </figure>
</p>
</div>

In MobileNet V2, the biggest difference is the introduction of the "bottleneck block". The bottleneck block adds a skip connection at the beginning of the next bottleneck block. Additionally, an expansion and projection are added in the bottleneck block. 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8eff937b-241e-4a69-b877-a1ed564efa7f" height="200" alt=""/>
    </figure>
</p>
</div>

In a bottleneck block, dimensions are jacked up so the network can learn a richer function. But they are projected down before moving the next bottleneck block so memory usage won't be too big.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/32afa29b-3146-4dd8-9dee-de772b633f70" height="200" alt=""/>
    </figure>
</p>
</div>
