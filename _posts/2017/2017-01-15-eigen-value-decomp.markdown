---
layout: post
title: Math - Eigen Value, Eigen Vector, and Eigen Value Decomposition
date: '2017-01-15 13:19'
subtitle: Covariance Matrix, PCA
comments: true
tags:
    - Math
---

## Eigen Values and Eigen Vectors

Basic Definitions

$$
\begin{gather*}
Ax = \lambda x
\end{gather*}
$$

- $x$ is an eigen vector, $\lambda$ is an eigen value

Important properties

- Only square matrices have eigen values and vectors

### How To Find Eigen Values and Eigen Vectors


1. Solve the characteristic equation:

$$
\begin{gather*}
\begin{aligned}
& det(A - \lambda I) = 0
\end{aligned}
\end{gather*}
$$

This gives eigen values $\lambda_1...$

2. For each eigen value:

$$
\begin{gather*}
\begin{aligned}
& (A - \lambda I) v = 0
\end{aligned}
\end{gather*}
$$

This is a singular system (whose determinant is 0). The system has non-trivial solutions. One can use Gaussian elimination to solve for v. 


## Eigen value Decomposition

Say a matrix $A$ has two eigen vectors $v_1$, $v_2$, and their corresponding eigen values are: $\sigma_1$, $\sigma_2$

Then, we have

$$
\begin{gather*}
A \begin{bmatrix}
v_1 & v_2
\end{bmatrix}
= 
\begin{bmatrix}
v_1 & v_2
\end{bmatrix}
\begin{bmatrix}
\lambda_1 & 0 \\
0 & \lambda_2
\end{bmatrix}
\end{gather*}
$$

So, we can get **Eigen Value Decomposition**:

$$
\begin{gather*}
V = \begin{bmatrix}
v_1 & v_2
\end{bmatrix},

\Lambda = \begin{bmatrix}
\lambda_1 & 0 \\
0 & \lambda_2
\end{bmatrix}

\\ =>
A = V \Lambda V^{-1}
\end{gather*}
$$

## Applications

### Series of self-multiplications

Assume we want to apply the same linear transform 8 times. Say, $A^8$

Matrix multiplication is expensive. One can use divide and conquer, and do the multiplication in the order of $log2(8)$ times.

But with Eigen Value Decomposition, this problem becomes: 

$$
\begin{gather*}
A^8 = V \Lambda^8 V^{-1}
\end{gather*}
$$

$\Lambda^8$ is easy to calculate, because it's just a diagonal matrix.

## Covariance Matrix

$$
\begin{gather*}
\begin{aligned}
& \Sigma = \frac{1}{N-1} (X-\mu) (X-\mu)^T
\end{aligned}
\end{gather*}
$$

In other words, $\Sigma = A^TA$ where A is a normalized and scaled $X$. $\Sigma$ is **symmetric, and positive semi-definite.**

The eigen vectors of $\Sigma$ are orthogonal. For an arbitrary eigen vector, $v$, there is $\Sigma v = \lambda v = A^TA v$. The eigen vector with the largest eigen value is the vector of the fitted line. Why?

$$
\begin{gather*}
\begin{aligned}
& v^T \Sigma v = v^T A^TA v = v^T \lambda v
\\ &
= (Av)^T (Av) = \lambda
\end{aligned}
\end{gather*}
$$

Note that each row in A is a normalized point `p` in X. So $Av$ is the **projection** of the vector `op` on the eigen vector, `v`. The largest $\lambda_m$ gives the eigen vector $v_m$ with the largest total projection. 


<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/39393023/429242062-f772fbf2-2d0a-48e4-b8df-9d0da1899354.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250401%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250401T224022Z&X-Amz-Expires=300&X-Amz-Signature=0f1d7c526fb64167c27584e34c487a67dac14b8b52a4d67c5d612b340d07f3cb&X-Amz-SignedHeaders=host" height="300" alt=""/>
            <figcaption><a href="https://zhuanlan.zhihu.com/p/435001757">Source: zhihu</a></figcaption>
       </figure>
    </p>
</div>