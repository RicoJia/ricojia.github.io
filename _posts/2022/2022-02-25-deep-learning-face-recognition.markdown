---
layout: post
title: Deep Learning - Face Recognition
date: '2022-02-25 13:19'
subtitle: Siamese Network, Deep Face
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction

Face verification (easier) vs face recognition (harder)

- Face verfifaction takes  input image, name and an ID. Then, it ouptuts if the image corresponds to the ID.
- Face recognition has a database K person's images. The input is an image, it outputs which person the image corresponds to.

After face recognition, liveness detection has been big in China.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/cb6f9998-394d-4433-bbcc-2951dc518d7d" height="300" alt=""/>
        <figcaption>If you are a picture, the system knows it!</figcaption>
    </figure>
</p>
</div>

Today's commercialized face recognition systems are trained on very large datasets. So this is one realm where transfer learning is nice.

## One-Shot Learning Problem

Conventional neural network `input -> CNN -> softmax(x)` is not scalable. One-shot learning allows you to have only one reference image. Then you can
compare that reference image against another input image. The essense to the one-shot learning problem is **learning the encodings of images, then learning a similarity function**.

$$
\begin{gather*}
d(img1, img2) = \tau
\end{gather*}
$$

### DeepFace

**Contribution:** DeepFace (2014 CVPR) focuses on **learning encodings of input frontal views of face images**.

**Method:** It does so by training a single convolutional neural network (CNN) to map each face image to their labels. They are connected to an FC layer that gets activated by softmax, and outputs class labels. The second last layer outputs embeddings (a 4096-vector), so during inference, this layer is often discarded.

In the deepface paper, they didn't use triplet to train. That's the work from FaceNet

Some key contributions from DeepFace:

- Max-pooling makes output of conv more robust to local translations. Then the output image will have activations shifted too

- "Several levels of pooling would cause the network to lose information about the precise position of detailed facial structure and micro-textures" I guess that's true
    - "Hence, we apply max-pooling only to the first convolutional layer. We interpret these first layers as a front-end adaptive pre-processing stage" - but this is not the case for resnet, yolo etc., right? So this is ok?

- The loss is the regular softmax-loss

### FaceNet

Why is the network called Siamese? Is that Thai? No, it came from "Siamese Twins", where you have two identical "twin networks" that share the same weights and same architecture. The twins just take in different inputs. Actually, It's a concept that was explored before DeepFace.

A **Siamese** network first learns the **encoding** of an image. This idea was introduced by DeepFace [1]. FaceNet learns directly from a face image, so it's end-to-end learning.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/cbf9712f-3a47-4bbd-ae80-a1e93633712f" height="300" alt=""/>
    </figure>
</p>
</div>

### Similarity Learning

After learning the encoding of two images, a similarity function can be formulated as a **triplet loss**:

#### Definitions

- Anchor: image of a person
- Positive: image of the same person
- Negative: image of a different person
- Triplet Loss is:

$$
\begin{gather*}
L(A, P, N) = max(|f(A) - f(P)|^2 - |f(A) - f(N)|^2 + \alpha, 0)
\end{gather*}
$$

Where $f(P)$ is the encoding of a "positive" image, and $f(N)$ is that of a negative image.
We append $\alpha$ to the loss function because we don't want the learned encoding is 0. This way, the model will learn an encoding such that distances between positive images are far smaller than those between negative images. **$\alpha$ is usually a hyperparameter (so not trainable)**

Triplet selection has to be careful too. We want to choose triplets `(image, true image, negative image)` that are hard to train on. Easy

### Triplet Selection

Some triplets already satisfy the loss. Why would it be a hassle to select them? There's no harm in incorporating them, but they could significantly slow down your training. To address this, Online Triplet Mining is employed within each mini-batch to select the most informative triplets that actively contribute to learning. Here's how it works:

1. Calculate embeddings of all images using the current model. (These will be computed in parallel in PyTorch/TensorFlow)
2. Compute each pair's similarity.
3. For each anchor, find the farthest distance positive image, and the closest distance negative image. Those are "hard triplets"
4. Total triplet loss is $\sum_m L(A, P, N) = \sum_m max(|f(A) - f(P)|^2 - |f(A) - f(N)|^2 + \alpha, 0)$ over batch $m$, or one can take the average as well.
5. Backpropagation starts with final output gradient:

$$
\begin{gather*}
\frac{\partial L}{\partial f(A)} = 2(f(A) - f(P)) - 2(f(A) - f(N)) = 2(f(N) - f(P))
\end{gather*}
$$

This can be handled by auto-diff and the gradient's computational graph.

#### Alternative 1: Binary Classification

The idea is to put two twin networks together, and try to have 1 neuron with weights $W$ and activation $\sigma(x)$ to learn if the two images are the same.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/8564259c-a447-4bb2-8bb8-58c369155186" height="300" alt=""/>
    </figure>
</p>
</div>

This would become:

$$
\begin{gather*}
\sigma(\sum_K w_k |f(x^i)_k - f(x^j)_k| + b)
\end{gather*}
$$

Here, the similarity of two images can also be [Chi-Squared Similarity](../2017/2017-06-05-math-distance-metrics.markdown). For anchor images, we can store the pre-computed value.

#### Implementation

1. Use a pretrained Inception Network to compute encodings of faces.
    - Input: `160x160` RGB images shaped into `(𝑚,160,160,3)`.
    - Output: `128-vector`. **This layer needs to be added to the original Inception Network**
        ```
        GoogLeNet: Convolutional Layers --> 1000-D FC Layer --> Softmax (Classification)
        FaceNet: Convolutional Layers --> 128-D FC Layer --> L2 Normalization (Embedding)
        ```
        - The output `128-vector` in the Inception network is a design trade off for small memory footprint and distinctiveness. In the original inception network, Szegedy et al. did NOT add an 128 output. The embeddings were also learned in the meantime.
    - Another difference between FaceNet and Inception Network is the FaceNet does NOT have an auxillary layer.
    - FaceNet also uses L2 norm in the triplet loss
    - Image Resizing should be handled here

2. Pass two faces through the Siamese Network.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/b666c15b-9a6f-4901-a9e3-d25119118e4f" height="300" alt=""/>
    </figure>
</p>
</div>

3. Triplet loss will try to push the encodings of two images of the same person (anchor and positive) close, while anchor and negative farther.

## References

[1] [Taigman, Y., Yang, M., Ranzato, M. A., & Wolf, L. 2014. DeepFace: Closing the Gap to Human-Level Performance in Face Verification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, 1701-1708. DOI: https://doi.org/10.1109/CVPR.2014.220](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)

[2] [Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 815-823. https://doi.org/10.1109/CVPR.2015.7298682](https://arxiv.org/pdf/1503.03832)

[3] [David Sandberg's Github Implementation for FaceNet](https://github.com/davidsandberg/facenet)
