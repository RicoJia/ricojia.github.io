---
layout: post
title: Deep Learning - Knowledge Distillation
date: '2022-05-21 13:19'
subtitle: Knowledge Distillation
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction To Knowledge Distillation

The goal of Knowledge Distillation is to train a small "student" network to mimic the output of a large "teacher" network. This is a.k.a "model compression"

If we are dealing with an image classification task, the output of the teacher models are probabilities across multiple classes. In that case, the teacher's output will be `[p_1, p_2, ...]` and this is NOT an one-hot vector. This is called **"soft labels"**.

### Implementation

#### Loss

It's important for the loss to be able to handle probabilities instead of hard class assignments.

- TensorFlow: `tf.keras.losses.CategoricalCrossentropy` (why?)
- PyTorch: `torch.nn.KLDivLoss()` [(Kullback-Leibler Divergence, or KL Divergence)](../2017/2017-06-05-math-distance-metrics.markdown)