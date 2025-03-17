---
layout: post
title: RGBD SLAM Bundle Adjustment Part 2
date: '2024-07-12 13:19'
subtitle: RGBD SLAM Backend, G2O
comments: true
header-img: "img/post-bg-unix-linux.jpg"
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

## G2O

`g2o` **(General Graph Optimization)** is a nonlinear least squares optimization framework designed for solving graph-based problems, commonly used in robotics and computer vision. It efficiently optimizes problems like SLAM (Simultaneous Localization and Mapping), Bundle Adjustment, and pose graph optimization by representing the problem as a factor graph.


## How g2o Works

g2o is a "General Least Squares" optimizer, meaning it can optimize any least squares (LS) problem that can be represented as a graph. The optimizer works by:

    1. Computing the gradient of the total cost function at each vertex by summing all constraints.
    2. Applying the Levenberg-Marquardt or Gauss-Newton algorithm to optimize parameters, aiming to find a local (and hopefully global) minimum.

In SLAM, a **vertex** must satisfy SE(3) constraints, which define the pose of a robot in 3D space. Some key vertex and edge types in `g2o/types/sba/types_six_dof_expmap.h` include:

- `VertexSE3Expmap`: Represents robot poses in SE(3) (translation + rotation).
- `VertexSBAPointXYZ`: Represents 3D landmarks (map points).
- `EdgeProjectXYZ2UV`: Represents the projection of a 3D landmark onto the image plane.

g2o is widely used in SLAM algorithms such as ORB-SLAM. [👉 Example: Bundle Adjustment Demo](https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/ba/ba_demo.cpp)

### G2O Optimization Steps

1. Compute Residual Error
    - Calls `computeError()` to compute the residual error `e(X)`, which represents the difference between expected and observed measurements.

2. Compute Jacobian
    - Calls `linearizeOplus()` to compute the Jacobian matrix J.
    This function can be overridden by the user to provide a custom Jacobian.
    - If no override is provided, automatic differentiation is used.
    - In a simple case where we have a consecutive pose constraint, the residual (error) is computed as:

    $$
    \begin{gather*}
    \begin{aligned}
    & e = x_j - x_i
    \end{aligned}
    \end{gather*}
    $$
    - The Jacobian is:

    $$
    \begin{gather*}
    \begin{aligned}
    & J = \frac{\partial e }{\partial x_j} - \frac{\partial e }{\partial x_i}
    \end{aligned}
    \end{gather*}
    $$

3. Construct Hessian Matrix
    $$
    \begin{gather*}
    \begin{aligned}
    & H = J^\top J
    \end{aligned}
    \end{gather*}
    $$

4. Solve for $\Delta X$ using Gauss-Newton or Levenberg-Marquardt

5. Update Vertices: Each vertex (pose or landmark) is updated as:

    $$
    \begin{gather*}
    \begin{aligned}
    & X \rightarrow X + \Delta X
    \end{aligned}
    \end{gather*}
    $$

### G2O Components

#### **Vertices (State Variables)**: 

Vertices represent the state to be optimized, such as robot poses or landmarks. Example: 

```cpp
class VertexSE2 : public g2o::BaseVertex<3, SE2> {
    void setToOriginImpl() override {
        _estimate = SE2();
    }
    void oplusImpl(const double* update) override {
        Eigen::Vector3d v(update);
        _estimate = SE2::exp(v) * _estimate;
    }
}
```

- `3`: Specifies degrees of freedom (DOF) for optimization `(x, y, θ)`.
- `SE2`: The underlying Lie group representation. The optimizer only updates these 3 elements, even if the internal state has a different structure (e.g., `Eigen::Vector<D>`).
- `setToOriginImpl()` → Resets the vertex to an identity pose. 
- `oplusImpl()` → Applies an update using Lie algebra exponential mapping. 

#### Edges (Constraints)

Edges define constraints between vertices, such as:

- Odometry constraints (connecting robot poses).
- Landmark constraints (projecting landmarks into camera frames).
- Loop closure constraints (closing a trajectory loop to reduce drift).

### Parallelization & Performance

g2o **does not natively support multi-threading**, but parallel execution can be achieved **using OpenMP.**

Example: Parallelizing Error Computation with OpenMP

```cpp
#pragma omp parallel for
for (size_t i = 0; i < edges.size(); ++i) {
    edges[i]->computeError();
}
```

Comparison with Other Optimizers

- 🔥 Ceres Solver (Google): Uses multi-threading and automatic differentiation.
- ⚡ GTSAM (Georgia Tech): Supports multi-threading with Intel TBB and factor graph-based optimization.

## Finally...

I have an implementation of [the `cv::SolvePnP` frontend and g2o backend on github](https://github.com/RicoJia/dream_cartographer/tree/main/rgbd_slam_rico)

<iframe width="560" height="315" src="https://www.youtube.com/embed/jCsX9R2aa-I?si=JEyQF3Gw1BrXfVxO" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>



## Q&A

- Are pose graph and factor graph the same thing?
    -  A pose graph is a special case of a factor graph. It consists of the robot's poses, and the edges represent spatial constraints (like relative transformations) between these poses.
    - a factor graph is a more general representation that can include various types of variables (e.g., robot poses, landmarks, sensor biases) and factors (which represent measurements or constraints between these variables).
- Does ROS2 SLAM Toolbox use g2o or Ceres? G2O.
