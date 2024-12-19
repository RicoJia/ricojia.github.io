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

For a refresher of MAP, please [see here](../2017/2017-02-13-math-MAP-MLE.markdown)

In a linear system,

$$
\begin{gather*}
\begin{aligned}
& x_k = A_k x_{k-1} + u_k + w_k
\\
& z_k = C_{k} x_{k} + v_{k}
\end{aligned}
\end{gather*}
$$

Here, we assume:

- $x_k$, $x_{k-1}$ are ground truth
- control noise $w_k \sim \mathcal{N}(0, Q)$,
- observation $v_k sim \mathcal{N}(0, R)$
- each state vector $x_k$ has a covariance matrix $P_k$

So according to the [linear transforms of Multivariate Gaussian Distribution](../2017/2017-02-16-math-MultiVariate-Distribution.markdown), we can write the joint distribution:

$$
\begin{gather*}
\begin{aligned}
& P(x_k | x_{k-1}) \sim \mathcal{N}(A_k x_{k-1}, A_k^T P_{k-1} A_k + Q)
\\
& P(z_k | x_k) \sim \mathcal{N}(C_k x_{k}, R)
\end{aligned}
\end{gather*}
$$

- Note that the noise covariance $z_k$ is independent from $x_k$, that's why we have $P(z_k | x_k) \sim \mathcal{N}(C_k x_{k}, R)$

In the MAP framework, we are interested in finding the posterior $\hat{x_k}$ (estimate of $x_k$) using [the log trick on multivariate Gaussian distribution](../2017/2017-02-13-math-MAP-MLE.markdown):

$$
\begin{gather*}
\begin{aligned}
& \hat{x_k}, P_k = argmax [\mathcal{N}(C_k x_{k}, R) \mathcal{N} (A_k x_{k-1}, A_k^T P_{k-1} A_k + Q)]
\\
& \text{using the log trick: }
\\
& \Rightarrow \hat{x_k}, P_k = argmin[J_k] = argmin [(z_k - C_k x_{k})^T R^{-1} (z_k - C_k x_{k}) + (x_k - A_k x_{k-1})^T (A_k^T P_{k-1} A_k + R)^{-1} (x_k - A_k x_{k-1})]
\end{aligned}
\end{gather*}
$$

Let

$$
\begin{gather*}
\begin{aligned}
& P_k^* = A_k^T P_{k-1} A_k + R
\\
& \Rightarrow argmin[J_k] = argmin [(z_k - C_k x_{k})^T R^{-1} (z_k - C_k x_{k}) + (x_k - A_k x_{k-1})^T (P_k^*)^{-1} (x_k - A_k x_{k-1})]
\end{aligned}
\end{gather*}
$$

Now, let's do the heavy lifting: differentiating w.r.t our variable of interest, $x_k$, and solve it while it's 0:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial J_k}{\partial x_k} = -2 C_k^T R^{-1} (z_k - C_k x_{k}) + 2(P_k^*)^{-1} (x_k - A_k x_{k-1}) = 0
\\
& \Rightarrow \mathbf{C}_k^\top \mathbf{R}^{-1} \mathbf{z}_k - \mathbf{C}_k^\top \mathbf{R}^{-1} \mathbf{C}_k \mathbf{x}_k + (\mathbf{P}_k^{*})^{-1} \mathbf{x}_k - (\mathbf{P}_k^{*})^{-1} \hat{\mathbf{x}}_k^{*} = 0

\\
& \Rightarrow (\mathbf{C}_k^\top \mathbf{R}^{-1} \mathbf{C}_k + (\mathbf{P}_k^{*})^{-1}) \mathbf{x}_k = \mathbf{C}_k^\top \mathbf{R}^{-1} \mathbf{z}_k + (\mathbf{P}_k^{*})^{-1} \hat{\mathbf{x}}_k^{*}

\\
& \Rightarrow \mathbf{x}_k = (\mathbf{C}_k^\top \mathbf{R}^{-1} \mathbf{C}_k + (\mathbf{P}_k^{*})^{-1})^{-1} (\mathbf{C}_k^\top \mathbf{R}^{-1} \mathbf{z}_k + (\mathbf{P}_k^{*})^{-1} \hat{\mathbf{x}}_k^{*})

\end{aligned}
\end{gather*}
$$

We are skipping another leavy lift: getting the covariance of $x_k$, $P_k$. If you are curious, the covariance of a [multivariate Gaussian distribution can be achieved by taking the Hessian of its negative log function.](../2017/2017-02-16-math-MultiVariate-Distribution.markdown)

$$
\begin{gather*}
\begin{aligned}
& P_k = ((P_k^{*})^{-1} + C_k^T R^{-1} C_k)^{-1}
\end{aligned}
\end{gather*}
$$

By using the [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)

$$
\begin{gather*}
\begin{aligned}
& P_k = P_k^{*} - P_k^{*} C_k^{T}(R^{-1} + C_k P_k^{*} C_k^T)^{-1} C_k P_k^{*}
\end{aligned}
\end{gather*}
$$

Let

$$
\begin{gather*}
\begin{aligned}
& K_k = P_k^{*} C_k^{T}(R^{-1} + C_k P_k^{*} C_k^T)
\\
& \rightarrow P_k = P_k^{*} - K_k C_k P_k^{*}
\end{aligned}
\end{gather*}
$$

$x_k$ can be simplified to:

$$
\begin{gather*}
\begin{aligned}
& \mathbf{x}_k = (P_k^{*} - K_k C_k P_k^{*}) (\mathbf{C}_k^\top \mathbf{R}^{-1} \mathbf{z}_k + (\mathbf{P}_k^{*})^{-1} \hat{\mathbf{x}}_k^{*})
\\
& = A_k x_{k-1} − K_k C_k A_k x_{k-1} + K_k z_k = (x^* - K_k C_k x^* +  K_k z_k)
\end{aligned}
\end{gather*}
$$

So we can see that we've derived the covariance matrix $P_k$, Kalman Gain $K_k$, and the final estimate $x_k$

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/d60f0567-3d93-4748-a2e7-894d8b5f123c" height="300" alt=""/>
       </figure>
    </p>
</div>

## EKF (Extended Kalman Filter)

EKF is applied where the system is non-linear:

$$
\begin{gather*}
\begin{cases}
    x_k = f(x_{k-1}, u_k) + \mathbf{w}_k, \quad k = 1, \dots, N \\
    z_k = h(x_k) + \mathbf{v}_k
\end{cases}
\end{gather*}
$$

Our important intermediate variables are:

$$
\begin{gather*}
\begin{aligned}
& P_k^* = A_k^T P_{k-1} A_k + R \Rightarrow  P_k^* = \frac{\partial f}{\partial x} P_{k-1} \frac{\partial f^T}{\partial x} + R

\\
& K_k = P_k^{*} C_k^{T}(R^{-1} + C_k P_k^{*} C_k^T) \Rightarrow K_k = P_k^{*} \frac{\partial h}{\partial x}^{T} (R^{-1} + \frac{\partial h}{\partial x} P_k^{*} \frac{\partial h}{\partial x}^T)

\end{aligned}
\end{gather*}
$$

Finally our updates are:

$$
\begin{gather*}
\begin{aligned}
& x_k = x_k^* + K_k(z_k - C_k x_k^*) \Rightarrow x_k = x_{k-1} + K(z_t - h(x_k^*))

\\
& P_k = P_k^{*} - K_k C_k P_k^{*} \Rightarrow P_k = P_k^{*} - K_k \frac{\partial h}{\partial x} P_k^{*}
\end{aligned}
\end{gather*}
$$

## References

<https://blog.yxwang.me/2018/07/robotics-slam-week2/>
