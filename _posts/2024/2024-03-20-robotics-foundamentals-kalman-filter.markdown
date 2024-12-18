---
layout: post
title: Robotics Fundamentals - Kalman Filter
date: '2024-03-15 13:19'
subtitle: Kalman Filter Framework, Proofs With Minimizing Error Covariance
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

1. When a new control update $u_k$ comes in, we can calculate a prediction of x, $x_k^*$, **the prediction covariance matrix, $P_k^*$**, and the Kalman gain $K_k$

$$
\begin{gather*}
& \begin{aligned}
x_k^* = A_{k} x_{k-1} + u_{k}  \tag{1}
\end{aligned}
\\
& \begin{aligned}
P_k^* = A_{k} P_{k-1} A_{k}^T + R_k \tag{2}
\end{aligned}
\\
& \begin{aligned}
K_k = P_k^* C_k^T (C_k P_k^* C_k^T + Q_k)^{-1}  \tag{3}
\end{aligned}
\end{gather*}
$$

- Note that in this step, the covariance matrix $P_k^*$ should increase from $P_{k-1}$, because we are updating with the control signal and we don't have the feedback from observation yet.

2. When a new observation comes in, we can calculate our **posteriors**. Note, here $P_k$ is the final estimate of the covariance matrix

$$
\begin{gather*}
\begin{aligned}
& x_k = x_k^* + K_k(z_k - C_k x_k^*)    \tag{4}
\end{aligned}
\\
\begin{aligned}
& P_k = (I - K_k C_k) P_k^*     \tag{5}
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

## Proof 1 - Minimize State Estimation Covariance From Ground Truth (Most Common Proof)

The goal of Kalman filter is to minimize the state estimation covariance from ground truth $x_{k, true}$:

$$
\begin{gather*}
\begin{aligned}
& min E[(x_{k, true} - x_{k})(x_{k, true} - x_{k})^T]
\\
& \text{Given: }
\\
& z_k = H_k x_{k, true} + v_k
\\
& \rightarrow
\\
& e_k = (x_{k, true} - x_{k}) = x_{k, true} - (x_k^* + K_k(z_k - C_k x_k^*))
\\
& = x_{k, true} - (x_k^* + K_k(H_k x_{k, true} + v_k - C_k x_k^*))
\\
& =  (I - K_k H_k)(x_{k, true} - x_k^*) - K_kv_k
\end{aligned}
\end{gather*}
$$

Then, the above can be written as:

$$
\begin{gather*}
\begin{aligned}
& E[(x_{k, true} - x_{k})(x_{k, true} - x_{k})^T] = E(e_ke_k^T)
\\
& = E[((I - K_k H_k)(x_{k, true} - x_k^*) - K_kv_k) ((I - K_k H_k)(x_{k, true} - x_k^*) - K_kv_k)^T]
\\
& = E[(I - K_k H_k)(x_{k, true} - x_k^*)(x_{k, true} - x_k^*)^T(I - K_k H_k)^T] + E[(K_kv_k)(K_kv_k)^T]
\\
& -E[(I - K_k H_k)(x_{k, true} - x_k^*)(K_kv_k)^T] - E[(K_kv_k)(x_{k, true} - x_k^*)^T(I - K_k H_k)^T]
\end{aligned}
\end{gather*}
$$

### General Form of $P_k$

Since $v_k$ and $x_k$ are independent, cross terms are zero. So we can derive the general form of P_k:

$$
\begin{gather*}
\begin{aligned}
& P_k := E[(x_{k, true} - x_{k})(x_{k, true} - x_{k})^T]
\\
& = E[(I - K_k H_k)(x_{k, true} - x_k^*)(x_{k, true} - x_k^*)^T(I - K_k H_k)^T] + E[(K_kv_k)(K_kv_k)^T]
\\
& = (I - K_k H_k)P_{k}^*(I - K_k H_k)^T + K_k R K_k^T
\end{aligned}
\end{gather*}
$$

### Prediction of covariance $P_{k}^*$

For equantion(2) **a-priori error covariance** is:

$$
\begin{gather*}
\begin{aligned}
& P_{k}^* := E[(x_{k, true} - x_k^*)(x_{k, true} - x_k^*)^T]
\\
& \text{define:}
\\
& e_k^* = (x_{k, true} - x_k) = A(x_{k-1, true} - x_{k-1}) + w_k
\\
& P_{k}^* = E[(A(x_{k-1, true} - x_{k-1}) + w_k) (A(x_{k-1, true} - x_{k-1}) + w_k)^T]
\\
& = E[A P_{k-1} A^T] + E[w_k w_k^T]
\\
& = A P_{k-1} A^T + Q
\end{aligned}
\end{gather*}
$$

### Kalman Gain $K_k$ and Covariance $P_k$

When the covariance $P_k$ is the smallest, its partial derivative w.r.t $K_k$ should be 0

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial P_k}{\partial K_k} = 0
\\
& \rightarrow
\\
& \frac{\partial}{\partial K_k} \left[ (I - K_k C_k) P_k^* (I - K_k C_k)^T \right] = -C_k P_k^* (I - K_k C_k)^T - (I - K_k C_k) P_k^* C_k^T
\\
&

\end{aligned}
\end{gather*}
$$

Also, the other derivative

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial}{\partial K_k} \left[ K_k R K_k^T \right] = K_k R + R K_k^T
\\
\end{aligned}
\end{gather*}
$$

Note that $R = R^T$, $P_k^* = (P_k^*)^T$, $P_k = (P_k)^T$

Combine both results:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial P_{k}}{\partial K_k} = 2(-C_k P_k^* + K_k C_k P_k^* C_k^T + K_k R) = 0
\\
& \rightarrow C_k P_k^* = K_k \left( C_k P_k^* C_k^T + R \right)
\\
& \rightarrow K_k = P_k^* C_k^T \left( C_k P_k^* C_k^T + R \right)^{-1}
\end{aligned}
\end{gather*}
$$

Without proof, when K_k is optimal, $P_k = (I - K_k C_k) P_k^*$.

## Proof 2 - MAP (Maximum A-Posteriori)
