---
layout: post
title: "[Point Cloud Compression] Draco"
date: 2026-03-02 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---
## Introduction

(GOOGLE? TODO) Draco can turn floating point attr like positions, normals, colors, and texture coords into  a more compact representation (TODO: embedding?), then encoding the remaning patterns. For meshes, it also compresses connectives (basically, triangles from vertices). For point clouds, it skips connectivity and focuses on the points.

1. given vertices, quantize attrs:
 1. turn floats into int grid. This is where some loss can happen
 2. fewer quantization bits usually means smaller output but less precision
  1. What is quantization bits? TODO?  Give me a small example
2. instead of storing raw points, draco predicts from neighbors (PREDICT?? HOW TODO?)
 1. it only stores small residual (difference actual - pred) TODO: isn't that also in (x,y,z), which is no shortening?
3. Entropy code the result
 1. TODO: Do you have a learned CDF? Where does your CDF come from
4.

 TODO: is it generally losslesss?

## Example

Suppose your input positions are:

```python
points = [
    [0.12, 0.18, 0.05],
    [0.15, 0.20, 0.05],
    [0.18, 0.21, 0.04],
    [0.80, 0.90, 0.10],
]
```

Imagine Draco uses a simple quantization grid with step size `0.01`. Then the points become integers like this:

```python
quantized = [
    [12, 18, 5],
    [15, 20, 5],
    [18, 21, 4],
    [80, 90, 10],
]
```

This already helps because int are shorter, and are available for entropy encoding. Draco stores enough transform in (TODO: what transform info?) so the decoder can approximately reconstruct the original floats later.  

Now imagine a very simple predictor: “predict the next point from the previous one.” Then instead of storing every quantized point directly, you store:

```python
stored = [
    [12, 18, 5],    # first point stored as-is
    [ 3,  2, 0],    # [15,20,5] - [12,18,5]
    [ 3,  1,-1],    # [18,21,4] - [15,20,5]
    [62, 69, 6],    # [80,90,10] - [18,21,4]
]
```
