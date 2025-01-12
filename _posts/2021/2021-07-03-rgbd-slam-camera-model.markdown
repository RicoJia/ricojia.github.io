---
layout: post
title: Computer Vision - Pinhole Camera Model
date: '2021-07-03 13:19'
subtitle: This Blog Shows How A Small Magic Peek Hole Captures The World
comments: true
tags:
    - Computer Vision
---

## Introduction

Camera is intriguing. There have been many different types with different types of lenses (such as fisheye, wide angle Lens). However, the most original (and the simplest) form of camera is the pinhole camera. This pinhole model is crucial for using a camera, as later mathematical models are built on top of it.

Each camera has a 3D world coordinate system, and:

- Optical Center:  the origin of the camera coordinate system
- Image Plane: the plane where the image is taken. It should be focal length away from the optical center $f$
- Principle Point: where the optical center lands on the image plane.
- A convention of the coordinate system placement is that the **z axis points to the image plane, x points right, and y points down**
- **An image always starts from the top left corner**, and **x or u represent columns, y or v represent row index** (it's a weird convention, I feel ya).

If we go back to the original [pinhole model](https://en.wikipedia.org/wiki/Pinhole_camera_model), we can find that the image is upside down. But for most robotics applications, this can be simplified into the illustration below, where an image is placed in in front of the camera optical center

<p align="center">
<img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/aa1eb110-f272-4939-b586-44eecae787ef" height="400"/>
</p>

## Mathematical Model

Mathematically, in a pinhole model, the relationship between a pixel $[u, v]$ and its corresponding 3D coordinates $[X, Y, s]$ ($s$ is depth) are:

$$
\begin{gather*}
u = f_x X/s + c_x
\\
v = f_y y/s + c_y
\end{gather*}
$$

$f_x$, $f_y$ already includes the scaling ratio from image plane to pixel coordinates, and focal length along $x$ and $y$ axes. $[c_x, c_y]$ are the pixel coordinate of the principal point.

Then this can be written as:

$$
K = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\\
=>
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ s \end{bmatrix}
\\
=>
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} =  K \begin{bmatrix} X \\ Y \\ s \end{bmatrix}
\\
$$

We call $K$ intrinsics. Specifically, we call below "canonical coordinates" of the point, which is equivalent to the **projection of the 3D point onto a plane that's unit distance away from the optical center**

$$
\begin{gather*}
\begin{bmatrix} X/s \\ Y/s \\ 1 \end{bmatrix}
\end{gather*}
$$

If the 3D coordinates is not in the camera frame, we need to apply an external homongenous transformation (extrinsics) to the point:

$$
\begin{gather*}
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} =  K T \begin{bmatrix} X \\ Y \\ s \end{bmatrix}
\end{gather*}
$$

Where:

- $T=[R | t]$
- $R$ is the $SO(3)$ rotation matrix
- $t$ is the Cartesian translation vector.

### Depth Camera Models

I was curious to check out my Intel D415's camera info and saw:

```bash
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [898.3310546875, 0.0, 630.9402465820312, 0.0, 898.3310546875, 345.5320739746094, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [898.3310546875, 0.0, 630.9402465820312, 0.0, 0.0, 898.3310546875, 345.5320739746094, 0.0, 0.0, 0.0, 1.0, 0.0]
```

- D is the distorsion matrix (same as above)
- K is the camera intrinsics
- R is the rectification matrix. It's used to align multiple cameras so their epipolar lines could align. In this case, it's identity, so no rectification is applied.
- P: Projection matrix. It's used for 3D points -> 2D transformation. Usually (and in this case) it's identical to K, so no other transformation is necessary here.

In ROS,

- The data is usually `uint16_t` because 65536 is large enough for a camera that measures up to 65m, with 1mm resolution
- `roslaunch realsense2_camera rs_camera.launch align_depth:=true`

## References
<https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html>
