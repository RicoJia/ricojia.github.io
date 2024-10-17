---
layout: post
title: Deep Learning - Introduction
date: '2022-01-04 13:19'
subtitle: Why Do We Even Need Deep Neuralnets? Data Partition, ML Ops, Data Normalization
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---


## Why Do We Need Deep Learning

Any bounded continuous function can be approximated by an arbitrarily large single layer. W hy? The idea is roughly that the linear combinations of activation function can compose a pulses, like a sigmoid, or 4 ReLus. Then, the pulses can compose an arbitrary function

<p align="center">
<img src="https://github.com/RicoJia/Machine_Learning/assets/39393023/d1020b0c-776f-47c5-971f-b673d27e587b" height="300" width="width"/>
<figcaption align="center">Pulses Can Approximate An Arbitrary Function</figcaption>
</p>

## ML Design Process

```text
Choose dataset -> Choose network architecture (number of layers, number of hidden units on a layer, activation function at each layer, etc.)
```

Nowadays, It's very hard to guess all these params right in one shot. Intuitions from one domain (speech recognition) likely doesn't transfer well to another. **So going over the above iterations is key.** That said, one exception is that ConvNet/ResNet from computer vision transferred to speech recognition well.

### ML Ops

Machine Learning Operations is similar to DevOps in general software engineering, is to quickly find the best model in a set of given hyper parameters, and deploy them. Also, changes in the machine learning model or data can be quickly tested.

### Data Normalization

When data come in different scales, say feature 1 ranges from $[-100, 100]$, another ranges from $[-1, 1]$, then the cost over these two features could be quite elongated along feature 1. Therefore, feature 2's gradient could be really small, and the same learning rate may not perform as well.

To make the cost function optimize faster:

1. "Shift to the center" - subtract out the mean from inputs
2. "variance scaling" - find the variance of data $\sigma$, then perform $x /= \sigma$. This sets the input data to have variance of 1.
 note is **apply the same mean and variance on training and test inputs.**. Otherwise, results could be different. 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/67600541-961e-4096-a656-747e608274f6" height="300" alt=""/>
    </figure>
</p>
</div>

## Remarks On Programming Frameworks

You might have heard of Caffe, Caffe2, PaddlePaddle, Keras, Theano, and TensorFlow, by the year 2024 however, Pytorch has become the most popular programming framework due to its **ease of use**, and speed.

One thing is I'd pay more attention to is its true "open-sourceness". In the software industry, some frameworks were once open-source, but later moved to proprietary cloud service by the company that controlled it. Some examples include: Elastic and Kibana, Redis, MongoDB

ONNX (o'nnex) (Open Neural Network Exchange) is an open source format for **deep learning and traditional AI** models. It defines an extensible computation graph model, operators, and data types. So models of different platforms can be converted between each other. E.g., TensorRT <-> TensorFlow <-> Pytorch. **Pitfall: Certain Platforms may not have the most updated versions of the model, so we need to upgrade/downgrade versioning.**

In machine frameworks, there is a "backend" and high level APIs

### Machine Learning Backend

In ML, low level operations are handled by a backend:

- Tensor (multidimensional matrix) manipulations such as convolution, matrix multiplication.
- Hardware Abstraction: CPU, GPU, TPU.
- Automatic Differentiation (partial derivatives)

Common Backend platforms include: TensorFlow (Google), PyTorch (Facebook), CNTK (Microsoft Cognitive Toolkit), Theano, JAX (Google's library for numerical computing).

### High level APIs

Keras was started out as a standalone project. As of TensorFlow 2.x, Keras has been fully integrated into TensorFlow and is now serving as the default.

Other High Level APIs include:
- PyTorch Lightning: wrapper around PyTorch so PyTorch Code could be more boiler plate; simplifies workflows such as for distributed training.
- Gluon (Apache MXNet)
- Sonnet (DeepMind)
- Flax (by Google For JAX).