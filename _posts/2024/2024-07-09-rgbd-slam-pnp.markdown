---
layout: post
title: RGBD SLAM - The PnP Problem
date: '2024-07-09 13:19'
subtitle: Solving the PnP problem - turning pixels into 3D positions! 
comments: true
tags:
    - RGBD Slam
---

## Intro

The Perspective-n-Point (PnP) problem is a very important technique in RGBD SLAM. In RGBD SLAM, it's quite common to see PnP as a front end , and bundle adjustment as the backend. In 2D-2D Methods, epipolar constraint is key for measuring the relative motion between two camera frames. In PnP, we are given the 3D coordinates of points (in world frame, and the camera frame), their 2D coordinates and, and matches. In this case, we have depth $z$, hence we do not need to apply epipolar constraints.

<p align="center">
<img src="https://github.com/RicoJia/The-Dream-Robot/assets/39393023/c52064b4-ddaf-40ed-974a-cf30dc0addb9" height="400" width="width"/>
</p>

### Direct Linear Transform

Naively, when we have the 3D coordinates of points in the camera frame and the world frame, we should be able to find a way to solve for their transforma, R and T.

$$
\begin{gather*}
K^{-T}z[u,v] = 

\begin{bmatrix}
t_1 & t_2 & t_3 & t_4 \\
t_5 & t_6 & t_7 & t_8 \\
t_9 & t_{10} & t_{11} & t_{12} \\
\end{bmatrix}

\end{gather*}
$$

So each point gives:
$$
\begin{gather*}
u_1 = \frac{t_1 X + t_2 Y + t_3 Z + t_4}{t_9 X + t_{10} Y + t_{11} Z + t_{12}}, \quad
v_1 = \frac{t_5 X + t_6 Y + t_7 Z + t_8}{t_9 X + t_{10} Y + t_{11} Z + t_{12}}.
\end{gather*}
$$

Let $P=[x,y,z]$

$$
\begin{gather*}
\mathbf{t}_1 = (t_1, t_2, t_3, t_4)^\top, \quad
\mathbf{t}_2 = (t_5, t_6, t_7, t_8)^\top, \quad
\mathbf{t}_3 = (t_9, t_{10}, t_{11}, t_{12})^\top
\end{gather*}
$$

Then, we gather at least 6 points, and use SVD to find a solution using least-squares for t.

$$
\begin{gather*}
\begin{pmatrix}
\mathbf{P}_1^\top & 0 & -u_1 \mathbf{P}_1^\top \\
0 & \mathbf{P}_1^\top & -v_1 \mathbf{P}_1^\top \\
\vdots & \vdots & \vdots \\
\mathbf{P}_N^\top & 0 & -u_N \mathbf{P}_N^\top \\
0 & \mathbf{P}_N^\top & -v_N \mathbf{P}_N^\top \\
\end{pmatrix}
\begin{pmatrix}
\mathbf{t}_1 \\
\mathbf{t}_2 \\
\mathbf{t}_3 \\
\end{pmatrix}
= 0
\end{gather*}
$$

Since we need $SE(3)$ constraints on R, we need to use QR decomposition to solve for R, while it's relatively simple to solve for t since it's in the Cartesian Space. Since we are getting an approximate solution from QR decomposition, we often need to optimize based on this solution.

### P3P

In the PnP set up, O is the origin of the camera frame, and we know the 3D points A, B, C in the world frame, after 2D feature matching. In the current camera view, we know their canonical coordinates, a, b, c. Our unknowns are $OA$, $OB$. $OC$ [1]. 

First we can solve for cosines:

$$
\cos(\alpha) = \frac{oa^2 + ob^2 - ab^2}{2 \cdot oa \cdot ob}
\\
\cos(\beta) = \frac{pa^2 + pc^2 - ac^2}{2 \cdot pa \cdot pc}
\\
\cos(\gamma) = \frac{pb^2 + pc^2 - bc^2}{2 \cdot pb \cdot pc}
$$

Then, using the law of cosines, we can 

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


Then, we need 1 pair of feature match to find the solution that yields the positive z.

One can notice that comOAred to the result of 8-point-algorithm, the rotation matrix is similar, but the translation is usually quite different

#### Disadvantages of P3P

- When there are more than 3 pairs of points, we cannot use them. **Question: can we use RANSAC?** TODO
- Sensitive to noise / feature mismatches.

## Implementation Notes

1. Open CV Error from `solvePnP`

```bash
what():  OpenCV(4.2.0) ../modules/calib3d/src/solvepnp.cpp:753: error: (-215:Assertion failed) ( (npoints >= 4) || (npoints == 3 && flags == SOLVEPNP_ITERATIVE && useExtrinsicGuess) ) && npoints == std::max(ipoints.checkVector(2, CV_32F), ipoints.checkVector(2, CV_64F)) in function 'solvePnPGeneric'
```

[In OpenCV](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html), there are implementations for the P3P [1], EPnP, etc.

2. Rviz does not show `sensor_msgs/PointCloud2` point cloud, even though there are valid messages. In this case, please check:

- Are you publishing on to a different frame? If so, is there a valid transform?
- Are there any `nan` or `inf` in your message? 

3. [How does OpenCV solves for extrinsics?](https://github.com/opencv/opencv/blob/f824db4803855ca30bf782f8bb37ca39051f319f/modules/calib3d/src/calibration.cpp#L923) `cvFindExtrinsicCameraParams2` is the function to look at. An excerpt of code include:

```cpp
CV_IMPL void cvFindExtrinsicCameraParams2(...){
    cvConvertPointsHomogeneous( objectPoints, matM );
    cvConvertPointsHomogeneous( imagePoints, _m );
    cvSVD( &_MM, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
    if (co-plannar-from-svd)
        cvFindHomography( _Mxy, _mn, &matH );
    else
        DLT();  //
        ...
    // minimizes reprojection error between two images
    CvLevMarq solver( 6, count*2, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,max_iter,FLT_EPSILON), true);
}
```

- One note is that the co-plane check using SVD is smart, [check here for more info](https://ricojia.github.io/2017/02/07/svd.html)

### References

[1] Complete Solution Classification for the Perspective-Three-Point Problem" by Xiao-Shan Gao, Xiao-Rong Hou, Jianliang Tang, and Hang-Fei Cheng. It was published in the IEEE Transactions on OAttern Analysis and Machine Intelligence, volume 25, issue 8, OAges 930-943, in 2003

[2] https://blog.csdn.net/leonardohaig/article/details/120756834

[3] Jesse Chen's Blog about EPnP https://blog.csdn.net/jessecw79/article/details/82945918