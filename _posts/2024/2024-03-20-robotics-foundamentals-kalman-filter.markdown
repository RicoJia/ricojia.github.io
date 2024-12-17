---
layout: post
title: Robotics Fundamentals - Kalman Filter
date: '2024-03-15 13:19'
subtitle: Kalman Filter Framework, Proofs From Multiple Perspectives
comments: true
tags:
    - Robotics
---

## Kalman Filter FrameWork -  Welcome to KF

SLAM is a state estimation problem. A typical state estimation framework in discrete time is:

$$
\begin{gather*}
\begin{cases}
    x_k = f(x_{k-1}, u_k) + \mathbf{w}_k, \quad k = 1, \dots, N \\
    z_k = h(x_k) + \mathbf{v}_k
\end{cases}
\end{gather*}
$$

Where

- f is the **state transition function**, and h is the **measurement (observation) function**.
- $v_k ~ N(0, Q_k)$ and $w_k ~ N(0, R_k)$ are Gaussian noises.
- $u_k$ is the "control signal", $z_k$ is the "observation".

When f and h are linear, that is, we have a linear system, our state estimation framework becomes:

$$
\begin{gather*}
\begin{cases}
    x_k = A_{k} x_{k-1} + u_{k} + w_{k} \\
    z_k = C_{k} x_{k} + v_{k}
\end{cases}
\end{gather*}
$$

Then, The framwork is as follows:

1. When a new control update $u_k$ comes in, we can calculate a prediction of x, $x_k^*$, **the covariance matrix, $P_k^*$**, and the Kalman gain $K_k$

$$
\begin{gather*}
\begin{aligned}
& x_k^* = A_{k} x_{k-1} + u_{k} \\
& P_k^* = A_{k} P_{k-1} A_{k}^T + R_k \\
& K_k = P_k^* C_k^T (C_k P_k^* C_k^T + Q_k)^{-1}
\end{aligned}
\end{gather*}
$$

- Note that in this step, the covariance matrix $P_k^*$ should increase from $P_{k-1}$, because we are updating with the control signal and we don't have the feedback from observation yet.

2. When a new observation comes in, we can calculate our **posteriors**:

$$
\begin{gather*}
\begin{aligned}
& x_k = x_k^* + K_k(z_k - C_k x_k^*)  \\
& P_k = (I - K_k C_k) P_k^*
\end{aligned}
\end{gather*}
$$

- Where $z_k - C_k x_k^*$ is called "innovation", meaning the "new information learned from the observation"

### Variable Dimensions

Vectors:

- $x_k$: `[m, 1]`
- $u_k$: `[m, m]`
- $z_k$: `[n, 1]`

Correspondingly, the matrices are:

- $A_k$: `[m, m]`
- $P_k$: `[m, m]`
- $C_k$: `[n, m]`
- $K_k$: `[m, n]`
- $R_k$: `[m, m]`
- $Q_k$: `[n. n]`

## Proof 1 - Minimize State Estimation Covariance (Most Common Proof in Textbooks)
