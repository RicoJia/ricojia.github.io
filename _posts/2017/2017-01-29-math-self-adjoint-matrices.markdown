---
layout: post
title: Math - Self-Adjoint Matrices
date: '2017-01-26 13:19'
subtitle: Robust Information Matrix
comments: true
tags:
    - Math
---

## Definition

A complex matrix is self adjoint if $A = \bar{A}^T$, reading "equal to its own conjugate transpose". It's also called "Hermitian". 

For real values, this means $A = A^T$

In Eigen, if you know $A = A^T$, you can use `Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);` to calculate its Eigen Vectors and Eigen Values. 

1. Reduces A to tridiagonal form (cheaper than the full Hessenberg Reduction that general matrices require)
2. Apply Symmetric QR or divide-and-conquer algorithm

It's roughly twice as fast on symmetric/Hermitian than a general solver.

An Eigen Solver uses an upper Hessenberg form, then runs the QR algorithm

PSD (Positive-Semi-Definite) matrix is a special type of **self-adjoint** matrix: 

$$
\begin{gather*}
\begin{aligned}
& A = A^T
\\ & x^T A x \ge 0
\end{aligned}
\end{gather*}
$$

## Robust Information Matrix

If we want to get the inverse of a self adjoint matrix robustly, first, we have

$$
\begin{gather*}
\begin{aligned}
 V = (U \Sigma U^T)
\\ \Rightarrow
V^{-1} = (U \Sigma U^T)^T = U \Sigma^{-1} U^T
\end{aligned}
\end{gather*}
$$

But we also want to floor the inverse values with an absolute floor value:

```cpp
Eigen::Matrix3d robustInfo(const Eigen::Matrix3d &cov,
                           double rel_floor = 1e-2,   // floor as % of σ_max
                           double abs_floor = 1e-4)   // or absolute floor
{
    // For symmetric PSD matrices Eigen's SelfAdjointEigenSolver is cheaper,
    // gives eigenvalues λ and eigenvectors V (cov = V Λ Vᵀ).
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    const auto &lambdas = es.eigenvalues();   // ascending order
    Eigen::Vector3d inv_lambda;

    const double λ_max     = lambdas.tail<1>()(0);
    const double floor_val = std::max(abs_floor, rel_floor * λ_max);

    for (int i = 0; i < 3; ++i)
        inv_lambda(i) = 1.0 / std::max(floor_val, lambdas(i));

    // info = V · diag(inv_lambda) · Vᵀ   (guaranteed symmetric PSD)
    return es.eigenvectors() * inv_lambda.asDiagonal() *
           es.eigenvectors().transpose();
}
```
