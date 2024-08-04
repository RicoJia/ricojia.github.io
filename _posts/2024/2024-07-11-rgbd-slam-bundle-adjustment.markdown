---
layout: post
title: RGBD SLAM - Bundle Adjustment From Zero To Hero, Part 1, Theories
date: '2024-07-11 13:19'
subtitle: RGBD SLAM Backend Introduction
comments: true
tags:
    - RGBD Slam
---

## What is Optimization?

$$
\begin{gather*}
min F(x)
\end{gather*}
$$

Above is an optimization problem. Any optimization needs: a cost function, target variables, and constraints. In SLAM, most problems are constraints-less, so we will be focused on it.

When F(x) is linear, we can form an optimization problem when it has constraints. When `F(x)` is a non-linear function, this optimization problem becomes a "non-linear" optimization problem (duh!). If we know the gradient of F(x) analytically, we can get a local minima in one-shot by $\frac{dF}{dx}=0$. However, in real life applications, we usually don't. So we would iteratively find the gradient descent direction in X, apply a step size, and get a smaller `F(x)`. To find the step size, we commonly use optimization techniques like Gauss-Newton, or Levenberg-Marquardt.

## How To Formulate SLAM Into An Optimization Problem

A typical SLAM frontend uses odometry (e.g., IMU) which is incremental and has accumulative errors. There are three main schools of SLAM methods: 

- Filtering Based SLAM (Kalman Filter, Particle Filter Based, etc.)
- Graph Based SLAM
- Deep Learning based SLAM

A batch is to optimize the total amount of errors with multiple camera frames at once.

Imagine we have a trajectory composed of:

- 4 poses in $[x, y, \theta]$
- 2 sythesized 2D observations $[d, \psi]$ ([range, bearing]). (Well, 3D has more dimension(s), but the formulation is the same.)

<div style="text-align: center;">
<p align="center">
    <figure>
    <img src="https://github.com/user-attachments/assets/c97fd3ef-0f9b-414a-8412-0d926f77c45a" height="300" alt=""/>
    </figure>
</p>
</div>

Now, we have 4 robot poses and 2 landmark poses to estimate. They are the variables under estimation, $X$. Now, we can represent the total error w.r.t all these variables:

- State vector: $[x_1, y_1, \theta_1 ... x_4, y_4, \theta_4]$
- Observations: $z = [ x_{z1}, y_{z1}, x_{z2}, y_{z2}]$
- Error $e_{ij}$: the difference between the estimated landmark pose $x_j$, and its observation from robot pose $x_i$
    $$
    \begin{gather*}
    e_{ij} =
    \begin{bmatrix}
    x_j - (x_i + d_i*cos(\theta_i + \psi_i)), \\
    y_j - (y_i + d_i*sin(\theta_i + \psi_i))
    \end{bmatrix}
    \end{gather*}
    $$

We can formulate our cost of this trajectory:

$$
\begin{gather*}
F(X) = \sum_{i \leq 6, j \leq 6} (e_{ij})^T \Omega e_{ij}
\end{gather*}
$$

- $\Omega$ is the **information matrix**, which measures the certainty of measurements. It's common to use $\Omega_{ij}=Q^{-1}$, where $Q$ is the covariance matrix of the observation model $[d, \psi]$

### How To Solve For $\Delta x$

$F(x)$ is a non-linear function, but its evaluation is a number. But it's a function of robot poses: $x_j$, $y_j$. So, **to find the set of poses that minimizes F(x)**, we can try to find its jacobian, and use the one of the above optimization techniques.

To be able to apply Gauss-Newton with increments on X, we need to linearize the cost function.

- For that, we first need to linearize error. We obtain Jacobian of error: $J = \frac{\partial e_{ij}}{\partial(X)}$

    - This will expand to $J = [\frac{\partial F}{\partial(x_1)}, \frac{\partial F}{\partial(y_1)}, \frac{\partial F}{\partial(\theta_1)} ... \frac{\partial F}{\partial(\theta_n)}]$. For a specific pair of nodes $(i, j)$, only F is only determined by $e_{ij}$. So
    
    $$
    \begin{gather*}
    J_{ij} = \begin{bmatrix}
    0_{2 \times 3} ... \frac{\partial e_{ij}}{\partial X_i}, ... 0_{2 \times 2} ... \frac{\partial e_{ij}}{\partial X_j} ...
    \end{bmatrix}
    \end{gather*}
    $$
    

Then, we given an initial set of estimate $X$, we want to apply a step size $\Delta X$ such that $F(X+\Delta X)$ is smaller. We can approximate the cost function locally with the first order taylor expansion, and get its local mimima 

$$
\begin{gather*}
F(X + \Delta X) = e(X+\Delta X)^T \Omega e(X+\Delta X)^T
\\ = (e(X) + J \Delta X)^T \Omega (e(X) + J \Delta X)
\\ = C + 2b \Delta X + \Delta X^T H \Delta X
\end{gather*}
$$

Where $H$ is the Hessian $J^T \Omega J$, and is composed of hessians from every robot-landmark pose pair:

$$
\begin{gather*}
H = \sum_{ij} H_{ij} = \sum_{ij} J^T_{ij} \Omega J_{ij} \text{(Gauss Newton)}
\\ \text{OR}
\\
H = \sum_{ij} H_{ij} = \sum_{ij} (J^T_{ij} \Omega J_{ij} + \lambda I) \text{(Levenberg-Marquardt)}
\end{gather*}
$$

The above is quadratic!! How nice. The minimum is achieved when

$$
\begin{gather*}
\Delta x = H_{ij}^{-1} b
\end{gather*}
$$

- To construct H, note that each block $H_{ij}$ is only a function of $x_i$ and $x_j$
- Correspondingly, $b = [... \frac{\partial e_{ij}}{\partial X_i}^T \Omega e_{ij} ... \frac{\partial e_{ij}}{\partial X_j}^T \Omega e_{ij} ...]$

### Solving H and J With Sparsity And Style

Now you might be wondering, why SLAM didn't do this pre-21st century? That is because solving for $H^{-1}$ was a real pain in the butt. It may have tens of thousands of edges, vertices, if not more.

The reason is that J and H are SPARSE, and we can use certain tricks to solve them more easily 

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/5e0be97f-2ccc-434d-86b1-04ea6646d830" height="300" alt=""/>
        <figcaption>Source: 14 Lectures on Visual SLAM </figcaption>
    </figure>
</p>
</div>

In $H=\sum_{ij} J^T_{ij} \Omega J_{ij}$, $\Omega$ is always a diagonal matrix. So, the $H_{ii}$ part is always diagonal. $H_{ij}$ and $H_{ji}$ may / may not be dense, depending on the observation data.

So, the final $J$ and $H$ are something like:

![Screenshot from 2024-07-29 20-18-31](https://github.com/user-attachments/assets/4efa33ff-5532-4de6-8c5f-4bc5384e63b1)


In a Larger system with hundreds of landmarks and robot poses, H may look like:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/a567b7ed-15cb-432a-a790-4e6e21bd9e64" height="300" alt=""/>
        <figcaption>Source: 14 Lectures on Visual SLAM </a></figcaption>
    </figure>
</p>
</div>


If we divide divide up $H$ into 4 parts:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/e7f447e6-f8a5-4a89-9d83-cb2e56a0351d" height="300" alt=""/>
        <figcaption>Source: 14 Lectures on Visual SLAM</figcaption>
    </figure>
</p>
</div>

Small bonus question here: why is C a diagonal matrix?

### Schur's Trick (or Schur Elimination)

$H \Delta X=b$ can be written as:

$$
\begin{gather*}
\begin{bmatrix}
B & E \\
E^T & C
\end{bmatrix}
\begin{bmatrix}
\Delta X_r \\ 
\Delta X_l
\end{bmatrix}
=
\begin{bmatrix}
v \\
w
\end{bmatrix}
\end{gather*}
$$

**It's much easier to invert a diaonal matrix**, because we just need to invert it's diagonal terms (C and without proof, B)

Then, by applying **Schur's compliment**: 

$$
\begin{gather*}
\begin{bmatrix}
I & -EC^{-1} \\
0 & I
\end{bmatrix}

\begin{bmatrix}
B & E \\
E^T & C
\end{bmatrix}
\begin{bmatrix}
\Delta X_r \\ 
\Delta X_l
\end{bmatrix}
=

\begin{bmatrix}
I & -EC^{-1} \\
0 & I
\end{bmatrix}
\begin{bmatrix}
v \\
w
\end{bmatrix}
\end{gather*}
$$

We get

$$
\begin{gather*}
\begin{bmatrix}
B -EC^{-1}E^T & 0 \\
E^T & C
\end{bmatrix}

\begin{bmatrix}
\Delta X_r \\ 
\Delta X_l
\end{bmatrix}
=


\begin{bmatrix}
v -EC^{-1}w\\
w
\end{bmatrix}
\end{gather*}
$$

So, we get a single equation for solving for $\Delta X_r$

$$
\begin{gather*}
(B -EC^{-1}E^T)^{-1} \Delta X_r = v -EC^{-1}w
\end{gather*}
$$

Note that $C$ is diagonal, so $C^{-1}$ is easy to get. $(B -EC^{-1}E^T)^{-1}$ is still something we need to brute-force invert, but its dimension is the same as the robot pose (which is much smaller than the original $H$). 

A side note about $S$'s sparsity: without proof, the an off-diagonal non-zero item $S_{mn}$ means there's at least 1 landmark observation between camera pose `m` and `n`. In general, we want $S$ to be dense, such that there will be constraints between camera poses to improve our estimates. In non-sliding window methods, such as ORB-SLAM, we may have a background thread running as the backend, so we could disgard frames that do not share many landmarks together.

### Solve for Delta x Using Cholesky Decomposition

After applying Schur's trick, we converted the original $H\Delta x = b$ into:
$$
\begin{gather*}
[B − EC^{−1} E^T] \Delta x_{r} = S \Delta x_{r} = v − EC^{-1} w = g'
\end{gather*}
$$

**$S$ is called "Schur's compliment"**. $S$ is still semi-positive-definite and symmetric. So, using Cholesky Decompistion, $S=LL^T$ where $L$ is a lower triangular matrix. So now,
1. Solve for $y$ in $Ly = g'$ because a lower triangular matrix's inverse is easier to solve
2. Solve for $\Delta x_{c}$ in $L^T \Delta x_{c} = y$

The system $Ax=b$ is always called "linear", **so its solver is called a "linear solver".**

### Wrap Up
After solving for $\Delta X_r$, one can use it to solve $\Delta X_p$. That gives the full step size $\Delta X$ for the optimizer.

ONEEEEEE LAST THINGGGGGG: wait a second, poses are in $SE(3)$.  $F(X + \Delta X) \approx F(X) + J\Delta X$ does NOT hold, especially the rotation matrix part in $SO(3)$. What do we do?? Well, the Lie Algebra of $SE(3)$, $se(3)$, DOES support addition:

$$
\begin{gather*}
\begin{bmatrix}
0 & -\omega_z & \omega_y & v_x \\
\omega_z & 0 & -\omega_x & v_y \\
-\omega_y & \omega_x & 0 & v_z \\
0 & 0 & 0 & 0
\end{bmatrix}
\end{gather*}
$$

The transformation between $se(3)$ and $SE(3)$ is called "matrix exponentiation", which is analogous to the regular scalar exponentiation / log operation.

If you want an interesting little nerdy story to share with your non-nerdy friends: $SE(3)$ is a manifold, and $se(3)$ is its tangent space. A manifold is a topological space that locally resembles Euclidean space and allows for calculus to be performed. (Here an $SE(3)$ can be transformed into a neary by $SE(3)$ smoothly through matrix multiplication)

