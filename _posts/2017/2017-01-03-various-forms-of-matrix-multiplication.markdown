---
layout: post
title: Math - Various Useful Forms Of Matrix Multiplication
date: '2017-01-05 13:19'
excerpt: Inner & outer Product, Correlation Matrix, etc.
comments: true
---


## Glossary

- Inner product: $<a,b> = \vec{a}^T \cdot \vec{b}$, which is a.k.a "dot product"
- Outer product: $a \otimes b = \vec{a} \cdot \vec{b}^T$, which results in a matrix.

## Matrix Multiplication And Outer Product

The definition of Matrix Multiplication of $C = AB$ is $C_{ij} = \sum_k A_{ik}B_{kj}$, where A is `mxn`, B is `nxp`

### The matrix product is the sum of the outer product of A's columns and B's rows

That is, $AB = \sum_{k=1}^n a_k b_k^{T}$. Why? Because for any given element $C_{ij}$, we have $C_{ij} = \sum_k A_{ik}B_{kj}$.

## Correlation Matrix

The Matrix $X^TX$ is called a correlation matrix of $X$. It is so very common in multiple fields, such as control system, SVD, etc. Each element is the inner product of `X_i` and `X_j^T`. And that's "correlation"

$$
\mathbf{X}^T \mathbf{X} =
\begin{bmatrix}
\mathbf{x}_1^T \\
\mathbf{x}_2^T \\
\vdots \\
\mathbf{x}_n^T
\end{bmatrix}
\begin{bmatrix}
\mathbf{x}_1 & \mathbf{x}_2 & \cdots & \mathbf{x}_m
\end{bmatrix}

\\=>

\begin{gather*}
\mathbf{X}^T \mathbf{X} =
\begin{bmatrix}
\mathbf{x}_1^T \mathbf{x}_1 & \mathbf{x}_1^T \mathbf{x}_2 & \cdots & \mathbf{x}_1^T \mathbf{x}_m \\
\mathbf{x}_2^T \mathbf{x}_1 & \mathbf{x}_2^T \mathbf{x}_2 & \cdots & \mathbf{x}_2^T \mathbf{x}_m \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{x}_m^T \mathbf{x}_1 & \mathbf{x}_m^T \mathbf{x}_2 & \cdots & \mathbf{x}_m^T \mathbf{x}_m \\
\end{bmatrix}
\end{gather*}
$$

So the correlation matrix is

- Positive semi-definite
- Symmetric