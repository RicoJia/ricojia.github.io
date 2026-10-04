---
layout: post
title: Math - Eigen Value, Eigen Vector, and Eigen Value Decomposition
date: 2017-01-15 13:19
subtitle: Covariance Matrix, PCA, Condition Number
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
---
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

### Zero Eigenvalues, Invertibility, and Condition Number

**A square matrix has a zero eigenvalue if and only if it is not invertible.** Proof: $\lambda = 0$ is an eigenvalue exactly when there is a nonzero $v$ with

$$
A v = 0 \cdot v = 0
$$

That means $A$ maps a nonzero vector to zero, so two different inputs ($v$ and $0$) give the same output, and $A$ cannot be undone. Equivalently, $\det(A) = \prod_i \lambda_i$, which is zero exactly when some $\lambda_i = 0$.

In matrix inverse, Eigen value decomposition shows what goes wrong numerically. Take a symmetric matrix $H$, such as a Hessian. **Its eigenvectors can be chosen orthonormal**, so $V^{-1} = V^T$ and

$$
\begin{gather*}
H = V \Lambda V^T, \quad
\Lambda = \begin{bmatrix}
\lambda_1 & & \\
& \ddots & \\
& & \lambda_n
\end{bmatrix}
\\
H^{-1} = V \Lambda^{-1} V^T, \quad
\Lambda^{-1} = \begin{bmatrix}
1/\lambda_1 & & \\
& \ddots & \\
& & 1/\lambda_n
\end{bmatrix}
\end{gather*}
$$

If some $\lambda_i = 0$, then $1/\lambda_i$ is undefined and $H^{-1}$ does not exist. If $\lambda_i$ is merely tiny, $H^{-1}$ exists but $1/\lambda_i$ is huge, and that causes instability.

**Example.** Solve $Hx = b$. Write $b$ in the eigenvector basis, $b = \sum_i \beta_i v_i$. Then

$$
x = H^{-1} b = \sum_i \frac{\beta_i}{\lambda_i} v_i
$$

Say one eigenvalue is $\lambda = 10^{-6}$ with eigenvector $v$, and $b$ has a tiny component along it: $b = 10^{-3} v + (\text{components along other eigenvectors})$. The component of $x$ along $v$ is then

$$
\frac{10^{-3}}{10^{-6}} = 10^{3}
$$

A component of $b$ that could easily be measurement noise gets amplified a million times, and it dominates $x$.

**Condition number.** This sensitivity is measured by the condition number. For a symmetric positive definite matrix,

$$
\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

(For a general matrix, use the ratio of the largest to smallest singular values instead.) A relative error in $b$ can be amplified by up to $\kappa$ in $x$:

$$
\frac{\|\delta x\|}{\|x\|} \le \kappa(H) \frac{\|\delta b\|}{\|b\|}
$$

As a rule of thumb, solving $Hx = b$ loses about $\log_{10} \kappa$ digits of precision. When $\lambda_{\min} = 0$, $\kappa = \infty$; see [the condition number of a plane's covariance matrix](https://ricojia.github.io/2017/02/25/math-plane-fitting/#condition-number-of-a-planes-covariance-matrix-is-infty) for an example. In SLAM, a tiny Hessian eigenvalue signals a poorly constrained direction; see [Hessian Degeneracy Test](https://ricojia.github.io/2026/05/14/Hessian-Degeneracy-Test/#small-eigenvalues).

---
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

## PCA

If a group of points form a plane, then its normal vector is the principal vector of the covariacne matrix. Other components, correspond to other columns of the covariance matrix? will be zero.
