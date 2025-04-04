---
layout: post
title: Robotics - [2D SLAM 1] Introduction and Scan Matching
date: '2024-04-10 13:19'
subtitle: Point-Point ICP, Point-Line ICP
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
    - SLAM
---

## Introduction

The assumption of a robot moving on a 2D plane is a strong one. However, many indoor robots do have support an assumption. Maps in this case can simply be interpreted as an image. Low-end 2D Lidars use a belt to turn its head. Hotel robots and cleaning robots usually use solid-state Lidars.

When the robots do need to perceive heights outside of its 2D Lidar's range, we usually either add an RGBD camera, or manually label on the map for restrcited areas.

- Labelling is more difficult in the 3D world. One example is labelling **heights of traffic lights**. In most cases, labelling is done on 2D.

In circa.2007, 2D SLAM was a hit. Sebastian Thrun and Gmapping authors: Wolfram Burgard, Cyrill Stachniss, and G Grisetti are some big names there. EKF SLAM family and the Particle SLAM Family (Fast-SLAM, RBPF SLAM) were popular. Their problem is that once the map is crooked, we can't easily fix them. 

Early days: no front end or back end; only one map is saved; Simple Loop closure detection;
Modern 2D SLAM: Pose Graph is used as backend; Key frames or sub-maps are the basic process units; Guarantees complete map after loop closure

| Early Days SLAM                           | Modern 2D SLAM                                      |
|-------------------------------------------|-----------------------------------------------------|
| No front end or back end                  | Pose Graph is used as backend                      |
| Only one map is saved                      | Key frames or sub-maps are the basic process units |
| Simple loop closure detection              | Guarantees complete map after loop closure         |


Concepts:
- Scan is a complete 360 deg LiDAR scan
- Scan matching is to find the relative pose between a scan vs another scan / map. **Scan matching is the core technique in a 2D SLAM**
- Submap: the map generated by several scans
- Occupancy grid: probability representation of a 2D map
    - Occupancy grid is far more common in robotics than in autonomous vehicles. Autonomous vehicles store vector-based road geometry, lane boundaries, traffic lights, and even crosswalks, and build a High-Definition Map. They need this precision to detect objects and track them.

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/5d26cec9-fef6-4cf0-8660-26abc3f9b3f5" height="300" alt=""/>
       </figure>
    </p>
</div>

## Scan Matching

The goal of scan matching is: given the observation model $z = h(x, u) + w$, find the most likely estimate 

$$
\begin{gather*}
\begin{aligned}
& X_{MLE} = argmax_{x}(x | z, m) = argmax_{x}(z | x, m)
\end{aligned}
\end{gather*}
$$

- Where `m` is thelast scan

To calculate which grid cells a ray goes through, we need ray casting (光线投射算法) and rasterization (栅格算法)

There are two main ways to do it:
- ICP (ICP, Point-to-Line ICP (PL_ICP), GICP, ICL): 2D and 3D are similar
- Likelihood fields (Gaussian Likelihood field, CSM)

The main problems in scan matching are:
- Which points should we perform scan matching on? 
    - All points for 2D
    - Sampling for 3D (according normal vectors, features, etc.)
- How to find point association? (KNN)
- How to find residuals. For scan point `[x,y]_i` and its matching point on another scan `[x, y]'_j`, we want to calculate its error (difference). The complete beam model is complicated and is not smooth for state estimation. We usually simplify the difference into a 2D Gaussian Distribution.

### Point-Point 2D Iterative Closest Point (ICP)

In 2D, robot pose is `[x, y, theta]`. The coordinate system we use are $T_{WB}$, and Later, sub map frame. A typical ICP-like algorithm iterative conducts 2 steps:

1. Data association
2. Pose estimation

#### [Step 1] Data Association

In this step, we use the KD-Tree to find the nearest neighbor of each point. Data association can be naively done using the nearest neighbors directly

#### [Step 2] Pose Estimation

A single 2D LiDAR Point comes in as $[\phi, r]$:

$$
\begin{gather*}
\begin{aligned}
& p_i^w = [x + r_i cos(\phi_i + \theta), y + r_i sin(\phi_i + \theta)]
\end{aligned}
\end{gather*}
$$

We can define **error as the cartesian difference between a specific point and its nearest neighbor in the other point cloud**:

$$
\begin{gather*}
\begin{aligned}
& e_i = p_i^w - q_i^w
\end{aligned}
\end{gather*}
$$

Then, the scan matching problem becomes a non-linear least-squares optimization problem:

$$
\begin{gather*}
\begin{aligned}
& (x, y, \theta)* = argmin(\sum_I |e_i|^2)
\end{aligned}
\end{gather*}
$$

This least squares problem is not linear (because of cos and sin in `p_i`), but it can be resolved by a Gauss-Newton minimizer like G2O. Therefore, we need the partial direvatives of each individual cost w.r.t `[x, y, theta]`:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial e_i}{\partial x} = [1, 0]
\\ &
\frac{\partial e_i}{\partial y} = [0, 1]
\\ &
\frac{\partial e_i}{\partial \theta} = [-r_i sin(\phi_i + \theta), r_i cos(\phi_i + \theta)]

\\ &
\Rightarrow
 \frac{\partial e_i}{\partial x} = \begin{bmatrix}
 1 & 0 & -r_i sin(\phi_i + \theta) \\
 0 & 1 & r_i cos(\phi_i + \theta) \\
 \end{bmatrix}

\end{aligned}
\end{gather*}
$$

Then, we can calculate $H$, $b$, and finally $x$ just like we do [for bundle adjustment](https://ricojia.github.io/2024/07/11/rgbd-slam-bundle-adjustment/)

$$
\begin{gather*}
\begin{aligned}
& H = \sum_{ij} H_{ij} = \sum_{ij} J^T_{ij} \Omega J_{ij} \text{(Gauss Newton)}
\\ &
b = J^Te
\\ &
\Rightarrow \Delta x = -H^{-1} b
\end{aligned}
\end{gather*}
$$

One advantage of 2D pose is its angles are additive, so it's not like 3D where we need to use `SO(3)` manifolds.

**Note that during optimization, after updating [x, y, theta], point correspondence would likely change**. So ICP is reliant on the initial relative pose estimate. In the below image, when the initial pose esimate is far off, point correspondence could be wrong:

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/e2ac93ed-086f-43eb-864f-7415be0ed571" height="300" alt=""/>
       </figure>
    </p>
</div>

### Point-Line 2D Iterative Closest Point (ICP)

Point-Line 2D ICP assumes that a point in the source cloud belongs to a line in the target point cloud. This could work well if the environment has linear structures like walls. 

#### [Step 1] Data Association

For a given point in the source cloud:

1. Find K nearest points in the target point cloud
2. Fit a line within those k points: `ax + by + c=0` using [Least squares](https://ricojia.github.io/2017/02/25/math-plane-fitting/)

#### [Step 2] Pose Estimation

1. Find the distance between the line and the point:

$$
\begin{gather*}
\begin{aligned}
& d = \frac{a p_x + b p_y + c}{\sqrt{a^2 + b^2 + c^2}}
\end{aligned}
\end{gather*}
$$

- If you are curious how to get the distance, hint: use the distance from the point to point (0, -c/b) and the Pythogorean Theorem. 

2. Because $\sqrt{a^2 + b^2 + c^2}$ is a constant, we use distance as error: `e = a p_x + b p_y + c`. The Jacobian is:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial e}{\partial p_x} = a
\\
& \frac{\partial e}{\partial p_y} = b
\\
\Rightarrow
\\
& \frac{\partial e}{\partial x} = \frac{\partial e}{\partial p} \frac{\partial p}{\partial x}
\\
& \frac{\partial p}{\partial x} = \begin{bmatrix}
1 & 0 & -r_i sin(\phi_i + \theta) \\
0 & 1 & r_i cos(\phi_i + \theta)
\end{bmatrix}
\\
\Rightarrow
\\
& \frac{\partial e}{\partial x} = \begin{bmatrix}
a & b & -a*r_i sin(\phi_i + \theta) + b * r_i cos(\phi_i + \theta)
\end{bmatrix}

\\
\Rightarrow
\\ &
J_i = e^\top e
\end{aligned}
\end{gather*}
$$

Then, we can calculate $H$, $b$, and finally $x$ just like we do [for bundle adjustment](https://ricojia.github.io/2024/07/11/rgbd-slam-bundle-adjustment/)

$$
\begin{gather*}
\begin{aligned}
& H = \sum_{i} H_{i} = \sum_{i} J^T_{i} \Omega J_{i} \text{(Gauss Newton)}
\\ &
b = \sum_{i} J_i^T e_i
\\ &
\Rightarrow \Delta x = -H^{-1} b
\end{aligned}
\end{gather*}
$$

## Likelihood Field Approach

ICP methods can be thought of as minimizing the "total potential energy" of the "springs" between point cloud 1 and 2. These springs however, need to be reinstalled at the beginning of each iteration. Likelihood field method can be thought of as a potential field. Different than a real physical potential field, our field has a resolution, and a maximum distance

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/989a5c99-9511-4030-bc3f-674d34ab7c1d" height="300" alt=""/>
       </figure>
    </p>
</div>

This likelihood field is also called "distance map". Around each point, the field will become weaker as distance goes up. 

For a given point $P^W$ in the world frame, we denote its field strength as $\pi (P^W)$. So the goal is to find the relative pose $x$ between point cloud 1 and 2 such that 

$$
\begin{gather*}
\begin{aligned}
& x^* = argmin(|\pi (P^W)|^2)
\end{aligned}
\end{gather*}
$$

To iteratively find that, we need the Jacobian of $\Pi$ w.r.t pose $x$:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \Pi}{\partial x} = \frac{\partial \Pi}{\partial P^W} \frac{\partial P^W}{ \partial x}
\end{aligned}
\end{gather*}
$$

Because in the distance field, we need to discretize the world coordinates with a resolution. So the relation between image coodinate and the world coordinate is:

$$
\begin{gather*}
\begin{aligned}
& P^f = \alpha P^W + c
\end{aligned}
\end{gather*}
$$

- $\alpha$ is the resolution, `c` is image center

So the Jacobian is finally:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \Pi}{\partial x} = \frac{\partial \Pi}{\partial P^f} \frac{\partial P^f}{\partial P^W}\frac{\partial P^W}{ \partial x}
\\
\Rightarrow 
\\ & 
\frac{\partial \Pi}{\partial P^W} = \alpha [\Delta \pi_x, \Delta \pi_y]
\\ & 
\Rightarrow 
\frac{\partial \Pi}{\partial x} = [\alpha \Delta \pi_x, \Delta \pi_y, -\alpha \Delta \pi_x r_i sin(\theta + \phi) + \alpha \Delta \pi_y r_i cos(\theta + \phi)]
\end{aligned}
\end{gather*}
$$

Where $[\Delta \pi_x \Delta \pi_y]$ is the image gradient

## Summary

Comparisons:

- Accuracies: NDT >= Likelihood field > PL-ICP > Point-Point ICP 
- Speed: Point-Point ICP ~= likelihood field > point to line

A skewed submap can lead to subsequent scans matching to incorrect features. Therefore, we introduce an "inlier" ratio of the $\Chi^2$ error that helps us identify not-so-good scans. However, if one uses multi-level scan matching, make sure there is a floor value for the threshold, because lower resolution maps may round down the threshold.

- This requires us to create debugging tools every step of the way. We need to make sure you can see numeric result and visualization at the same iteration (score, scan match)

Another thing is to use **bilinear interpolation** on likelihood field cells for better estimating the distance to obstacles if the inputs are float:

```cpp
template <typename T>
inline float get_bilinear_interpolated_pixel_value(const cv::Mat& img, float x, float y){
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    // Assuming this image is stored in contiguous image.
    // this gives pointer to (y,x).
    // data[1] is [y, x+1]. data[img.step / sizeof(T)] is [y+1, x],
    // and data[img.step / sizeof(T) + 1] is [y+1, x+1]
    const T* data = &img.at<T>(floor(y), floor(x));
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float((1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] + (1 - xx) * yy * data[img.step / sizeof(T)] +
                 xx * yy * data[img.step / sizeof(T) + 1]);
}
```
