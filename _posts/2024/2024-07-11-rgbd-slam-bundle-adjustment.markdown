---
layout: post
title: RGBD SLAM Bundle Adjustment
date: '2024-07-11 13:19'
excerpt: RGBD SLAM Backend Introduction
comments: true
---

## Misc Notes

A classical SLAM frontend is like IMU, it is incremental, and have accumulative errors. A batch is to optimize the total amount of errors with multiple camera frames at once.

## Problem Setup

There are three types of bundle adjustment:

- Feature graph
    - Each vertex is a camera pose / 3D point (feature)
    - Each edge is an edge that connects the observed 3D point to its camera pose. It represents the 2D reprojection error.
- Pose graph
    - each graph is a camera pose only
- Factor graph
    - TODO


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

- One reason for degredation is probably the number of matching feature points
    - if there are too few, we might want to adjust
- I think there are two questions if we want to use g2o:
    - the front end gives the transform from one frame to the next keyframe, right? If that's the case, then how do we detect if the same points show on multiple frame? And at different frames, the same feature points could correspond to different 3D poses?

### What is a kernel?

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


## G2O Introduction

G2O has a list of the optimizations it does. Here is short excerpt of the items that pertain to our rgbd slam problem:

- slam3d: 
    - Each vertex represents
        - a robot pose with 6dof.
        - 3D points (landmarks)
    - Each constraint represents
        -  the pose-pose constraint.
- SBA (Sparse Bundle Adjustment):
    - Each vertex represents:
        - camera intrinsics (optional?)
        - extrinsics
        - 3D points (landmarks)???
    - Each constraint include:
        - 2D projection (with known intrinsics?) onto image plane (2D coordinates)
        - monocular projection with parameters (with unknown intrinsics?)
        - stereo projection (3D points can be projected back onto the left and right cameras. The baseline between the cameras shhould be known)
        - scale constraint between extrinsics nodes? (Used in scenario where additional information about the relative scale / distance between multiple cameras is known)

Linear solvers include:
- PCG
- colamd
- CHOLMOD
- csparse
- dense
- eigen

## Schur's Trick

TODO

## Solve for Delta x Using Cholesky Decomposition

cfter applying Schur's trick, we get converted the original $H\Delta x = g$ into:
$$
\begin{gather*}
[B − EC^{−1} E^T] \Delta x_{c} = H' \Delta x_{c} = v − EC^{-1} w = g'
\end{gather*}
$$

**$H'$ is called "Schur's compliment"**. $H'$ is still semi-positive-definite and symmetric. So, using Cholesky Decomp, $H=LL^T$ where $L$ is a lower triangular matrix. So now,
1. Solve for $y$ in $Ly = g'$ because a lower triangular matrix's inverse is easier to solve
2. Solve for $\Delta x_{c}$ in $L^T \Delta x_{c} = y$

The system $Ax=b$ is always called "linear". Cholesky Decomp. finally solves this step, so its solver is called a "linear solver".

## How G2O Works

### Optimizers

Most SLAM problems are sparse, meaning they have a large number of variables, but relatively low number of edges. The SparseOptimizer is the most common type of optimizer used in SLAM. It first needs a block solver for computing Hessian jacobian, and Schur compliment in SparseBlockMatrix, a datastructure that represents sparse matrices with non-zero blocks only. Then, Sparse Optimizer needs a linear optimizer

- CSparse: using QR Factorization and LU factorization
- CHOLMOD: uses Cholesky Decomp. and is optimized for symmetric positive definite sparse amtrices.
- PCG (Preconditioned Conjugate Gradient) TODO?

SparseOptimizerIncremental: In online slam, optimization is done incrementally? TODO

- Marginalization TODO


## G2O Set Up

For least-squares problem

A **vertex** is the sets of parameters to optimize. In the context of slam, a vertex is the camera pose, i.e., $se(3)$ parameters (6 parameters). **A constraint, or an edge** is a measurement that was seen at least two camera poses, so in the graph, an edge connects at least two nodes.
Node error function?

Then, an optimizer's job is to 
- Find gradient of the total cost function at their vertices (i.e., adding up all constraints)
- Apply Levenberg-Marquardt on parameters under optimization to find a local (hopefully global) minimum.

In SLAM, a vertex needs to satisfy $se(3)$ requirements. In `g2o/types/sba/types_six_dof_expmap.h` these vertices can be defined:
- `VertexSE3Expmap` represents robot poses in SE3 space, 
- `VertexSBAPointXYZ` represents a 3D point
- `EdgeProjectXYZ2UV` represents the projection of a 3D point onto the image plane

G2O is used in famous SLAM algorithms like ORB_SLAM. 

[Example](https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/ba/ba_demo.cpp)

## References

[1] Fan Zheng's Farm: https://fzheng.me/2016/03/15/g2o-demo/
[2] g2o "what is in these directories": https://github.com/RainerKuemmerle/g2o/blob/master/g2o/what_is_in_these_directories.txt

