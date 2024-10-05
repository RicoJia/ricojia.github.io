---
layout: post
title: Deep Learning - Face Recognition
date: '2022-02-25 13:19'
subtitle: Siamese Network
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

## One-Shot Learning Problem

Conventional neural network `input -> CNN -> softmax(x)` is not scalable. One-shot learning allows you to have only one reference image. Then you can
compare that reference image against another input image. The essense to the one-shot learning problem is **learning the encodings of images, then learning a similarity function**.

$$
\begin{gather*}
d(img1, img2) = \tau
\end{gather*}
$$

### DeepFace


DeepFace focuses on **learning encodings of input frontal views of face images**. It does so by training a single convolutional neural network (CNN) to map each face image to their labels. They are connected to an FC layer that gets activated by softmax, and outputs class labels. The second last layer outputs embeddings (a 4096-vector), so during inference, this layer is often discarded.

In the deepface paper, they didn't use triplet to train. That's the work from FaceNet


### Encoding Learning

Why is the network called Siamese? Is that Thai? No, it came from "Siamese Twins", where you have two identical "twin networks" that share the same weights and same architecture. The twins just take in different inputs. Actually, It's a concept that was explored before DeepFace.

A **Siamese** network first learns the **encoding** of an image. This idea was introduced by DeepFace [1]. 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/cbf9712f-3a47-4bbd-ae80-a1e93633712f" height="300" alt=""/>
    </figure>
</p>
</div>

### Similarity Learning

After learning the encoding of two images, a similarity function can be formulated as a **triplet loss**:

$$
\begin{gather*}
L(A, P, N) = max(|f(A) - f(P)|^2 - |f(A) - f(N)|^2 + \alpha, 0)
\end{gather*}
$$

Where $f(P)$ is the encoding of a "positive" image (image of the same person), and $f(N)$ is that of a negative image.
We append $\alpha$ to the loss function because we don't want the learned encoding is 0. This way, the model will learn an encoding such that distances between positive images are far smaller than those between negative images.

Triplet selection has to be careful too. We want to choose triplets `(image, true image, negative image)` that are hard to train on. Easy

## References

[1] [Taigman, Y., Yang, M., Ranzato, M. A., & Wolf, L. 2014. DeepFace: Closing the Gap to Human-Level Performance in Face Verification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, 1701-1708. DOI: https://doi.org/10.1109/CVPR.2014.220](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)

