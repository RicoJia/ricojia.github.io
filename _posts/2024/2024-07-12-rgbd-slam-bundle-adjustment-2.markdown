---
layout: post
title: RGBD SLAM Bundle Adjustment Part 2
date: '2024-07-12 13:19'
subtitle: RGBD SLAM Backend Introduction
comments: true
tags:
    - RGBD Slam
---

If you haven't, please check out the previous article on [how to formulate SLAM as an optimization problem](./2024-07-11-rgbd-slam-bundle-adjustment.markdown)

## Why Graph Optimization

**Frontend and Backend**: the front end gives the transform from one frame to the next keyframe. The backend perrforms batch optimization, which takes multiple frames and observations, and try to minimize the total squared errors of estimated poses against observations.

**Motivation:** A classical SLAM frontend is like IMU, it is incremental, and have accumulative errors.  **The goal of graph optimization** is we want to minimize the total squared error of a cost function that minimizes observation errors, see here [See here](./2024-07-11-rgbd-slam-bundle-adjustment.markdown). Then, we can use the sparsity of the Hessian and Jacobi matrices to achieve that. **However, setting up the hessian and jacobi is manual, and the following optimziation process is quite "standard".**

So, we introduce graph optimization to add blocks to those matrices for those poses in the form of a graph, with nodes and edges.

## How To Formulate SLAM Optimization Into A Graph

What are nodes: each node represents a pose of the robot's trajectory. Each edge between two edges represent the relative position of the two poses.

A graph G is composed of vertices (V) and edges (E) $G={V,E}$. An edge can connect to 1 vertex (unary edge), 2 vertices (binary edge), or even multiple vertices (hyper edge). Most commonly, a graph has binary edges. But when there are hyper edges, this graph is called "hyper graph".

Let's look at an example:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/88b87054-fce9-4304-be4f-c6b595ff36ce" height="400" alt=""/>
        <figcaption>Source: 14 Lectures on Visual SLAM</figcaption>
    </figure>
</p>
</div>

We can form a graph with two types of nodes: 4 camera poses and 2 camera poses:

- $x_1$ is 1 set of 6 parameters for $SE(3)$ Pose $T_{x1}$
- $p_1$ is 1 set of 3 parameters for $(x,y,z)$ Pose $T_{p1}$

Then, we have two types of edges that represent **an error**:

- $x_1x_2$ is an adjacent edge with odometry observation $\hat{T_{x1,x2}}$ . It represents $T_{x1. x2} - \hat{T_{x1,x2}}$ in $se(3)$
- $x_1p_1$ is an observation edge (TODO is that right?) with observation $\hat{T_{x1,p_1}}$. For estimate $T_{p1}$, we can get error $T_{x1, p1} - \hat{T_{x1,p_1}}$. 


## How G2O Works

G2O is a "General Least Squares" Optimizer, meaning any LS problem that can be formulated in the form of a graph could be optimized like this. Then, an optimizer's job is to:

- Find gradient of the total cost function at their vertices (i.e., adding up all constraints)
- Apply Levenberg-Marquardt on parameters under optimization to find a local (hopefully global) minimum.

In SLAM, a vertex needs to satisfy $se(3)$ requirements. In `g2o/types/sba/types_six_dof_expmap.h` these vertices can be defined:
- `VertexSE3Expmap` represents robot poses in SE3 space, 
- `VertexSBAPointXYZ` represents a 3D point
- `EdgeProjectXYZ2UV` represents the projection of a 3D point onto the image plane

G2O is used in famous SLAM algorithms like ORB_SLAM. [Example](https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/ba/ba_demo.cpp)

### Optimizers

Most SLAM problems are sparse, meaning they have a large number of variables, but relatively low number of edges. The SparseOptimizer is the most common type of optimizer used in SLAM. It first needs a block solver for computing Hessian jacobian, and Schur compliment in SparseBlockMatrix, a datastructure that represents sparse matrices with non-zero blocks only. Then, Sparse Optimizer needs a linear optimizer

- CSparse: using QR Factorization and LU factorization
- CHOLMOD: uses Cholesky Decomp. and is optimized for symmetric positive definite sparse amtrices.
- PCG (Preconditioned Conjugate Gradient) TODO?

SparseOptimizerIncremental: In online slam, optimization is done incrementally? TODO

## Robust Kernels

In SLAM, it's common to have mismatched feature points. In that case, we add an edge that we shouldn't have added between a camera pose and a 3D point. The wrong edge could give huge error and high gradient, so high that just optimizing parameters associated could yield more gradient than the correct edges. One way to make up for it is to regulate an edge's cost, so it doesn't get too high nor gives too high of a gradient. Huber loss, cauchy loss are common examples. In SLAM terminology, loss is also called  "kernel". (How come people in different computer science disciplines love corn so much?)

For example, Huber kernel switches to first order when error is higher than $\delta$. Also, it is continuous and derivable at $y - \hat{y} = \delta$. This is important, because we need to get gradient everywhere at the cost function.

$$
\begin{gather*}
e=
\begin{cases} 
\frac{1}{2} x^2 & \text{for } |x| \le \delta \\
\delta (|x| - \frac{1}{2} \delta) & \text{for } |x| > \delta 
\end{cases}
\end{gather*}
$$

### Implementation

In general, a SLAM framework is:
```
def add_to_graph(rgb_frame, depth_frame, previous_rgb_frame, previous_depth_frame)
    matches = feature_matching(rgb_frame, previous_rgb_frame)
    if length(matches) < LEN_THRESHOLD:
        return
    estimate = pnp_estimate(matches, previous_depth_frame)
    # This is likely a bad estimate
    if estimate.linear_norm() > LINEAR_MAX or estimate.angular_norm() > ANGULAR_MAX:
        return
    # Not enough motion, reject
    if estimate.linear_norm() < LINEAR_MIN and estimate.angular_norm() < ANGULAR_MIN:
        return

    add_to_optimizer(rgb_frame)

def SLAM_pipeline(rgb_frame, depth_frame):
    add_to_graph(rgb_frame, previous_rgb_frame[-1]) 
    # back end
    # Proximity check
    for last N previous_rgb_frames:
        add_to_graph(rgb_frame, previous_rgb_frame[-n]) 

    # Random frame check for loop closer
    for M random previous_rgb_frames:
        add_to_graph(rgb_frame, previous_rgb_frame[-m]) 

    optimize() 
    
```


I have an implementation of [the `cv::SolvePnP` frontend and g2o backend on github](https://github.com/RicoJia/dream_cartographer/tree/main/rgbd_slam_rico)

<iframe width="560" height="315" src="https://www.youtube.com/embed/jCsX9R2aa-I?si=JEyQF3Gw1BrXfVxO" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
