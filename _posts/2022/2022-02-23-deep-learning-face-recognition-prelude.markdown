---
layout: post
title: Deep Learning - Face Recognition Prelude
date: '2022-02-23 13:19'
subtitle: Face Frontalization
comments: true
header-img: "img/home-bg-art.jpg"
tags:
    - Deep Learning
---

## Introduction

DeepFace introduced a 3D alignment step that projects 2D face images into a frontal view [1]. This is called "frontalization". A very cool 2D->3D problem. A frontal view is a view right in front of the face. When given a side view of the face, the face is warped. Some main methods include:

### 2D Frontalization

- Pre-define 6 fiducials points (anchor points) of a frontal face.
    - centers of the eye (2 points)
    - Tip of the nose, etc (1 point)
    - Corners of the mouth (2 points)
    - Center of the mouth (1 point)
- Detects 6 fiducial points (source points)
- To find the final cropped image, from these 6 points, generate an estimate transformation `[scale, rotation, and translation]`, apply it, detect the 6 points again, and repeat until no significant change detected in the 6 points' locations.
- [1] stated that 2D frontalization is bad for out-of-plane rotation (pitch and yaw)

The estimation is:
$$
\begin{gather*}
x_j^{anchor} = sR x_J^{source} + t_x
y_j^{anchor} = sR y_J^{source} + t_y
...
\end{gather*}
$$

We have 12 linear equations, so we can find parameters using Least-Squares. (Or RANSAC if we have more.). OpenCV's `cv2.estimateAffinePartial2D` can do this 

### 3D Alignment

The general process is:

- First creates an 3D model of average of a large 3D face database
- Try to map the 2D fiducials on the image onto the 3D model. That is:

$$
\begin{gather*}
s [u, v, 1] = K[R|t] [X, Y, Z, 1]
\end{gather*}
$$

- K is assumed to be identity if it's not known

This is a 2D-3D problem and we can formulate it as [a PnP problem](https://ricojia.github.io/2024/07/09/rgbd-slam-pnp/)


[1] [Taigman, Y., Yang, M., Ranzato, M. A., & Wolf, L. 2014. DeepFace: Closing the Gap to Human-Level Performance in Face Verification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, 1701-1708. DOI: https://doi.org/10.1109/CVPR.2014.220](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)

