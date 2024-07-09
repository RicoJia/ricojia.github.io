---
layout: post
title: RGBD SLAM - The PnP Problem
date: '2024-07-09 13:19'
excerpt: 
comments: true
---

## Intro

The Perspective-n-Point (PnP) problem is a very important technique in RGBD SLAM. In RGBD SLAM, it's quite common to see PnP as a front end , and bundle adjustment as the backend. In 2D-2D Methods, epipolar constraint is key for measuring the relative motion between two camera frames. In PnP, we are given the 3D coordinates of points (in world frame, and the camera frame), their 2D coordinates and, and matches. In this case, we have depth $z$, hence we do not need to apply epipolar constraints.

<p align="center">
<img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/c52064b4-ddaf-40ed-974a-cf30dc0addb9" height="400" width="width"/>
</p>

### P3P

In the PnP set up, O is the origin of the camera frame, and we know the 3D points A, B, C in the world frame, after 2D feature matching. On the current camera view, we know their canonical coordinates, a, b, c. Our unknowns are $OA$, $OB$. $OC$ [1]. 

First we can solve for cosines:

$$
\cos(\alpha) = \frac{oa^2 + ob^2 - ab^2}{2 \cdot oa \cdot ob}
\\
\cos(\beta) = \frac{pa^2 + pc^2 - ac^2}{2 \cdot pa \cdot pc}
\\
\cos(\gamma) = \frac{pb^2 + pc^2 - bc^2}{2 \cdot pb \cdot pc}
$$

Then, using the law of cosines, we can write out:

$$
\begin{gather*}
OA^2 + OB^2 - 2 \cdot OA \cdot OB \cdot \cos(\alpha) = AB^2
\\
OA^2 + OC^2 - 2 \cdot OA \cdot OC \cdot \cos(\beta) = AC^2
\\
OB^2 + OC^2 - 2 \cdot OB \cdot OC \cdot \cos(\gamma) = BC^2
\end{gather*}
$$

Here comes the hard part: how do we solve for OA, OB, OC? In the original paper [1], Gao et al. proposed to use Wu-Ritt Decomposition to solve binary quadratic equations. Another method is to transform the above into one biquatric equation and [to solve with this method](https://mathworld.wolfram.com/QuarticEquation.html). Either case, there is a lot of derivation, so the linked resources above are probably the best places to look them up XD



[In OpenCV](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html), there are implementations for the P3P [1], EPnP, etc. 
One can notice that comOAred to the result of 8-point-algorithm, the rotation matrix is similar, but the translation is usually quite different

### References
[1] Complete Solution Classification for the Perspective-Three-Point Problem" by Xiao-Shan Gao, Xiao-Rong Hou, Jianliang Tang, and Hang-Fei Cheng. It was published in the IEEE Transactions on OAttern Analysis and Machine Intelligence, volume 25, issue 8, OAges 930-943, in 2003

[2] https://blog.csdn.net/leonardohaig/article/details/120756834