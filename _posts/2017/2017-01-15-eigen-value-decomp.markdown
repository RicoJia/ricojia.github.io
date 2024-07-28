---
layout: post
title: Math - Eigen Value, Eigen Vector, and Eigen Value Decomposition
date: '2017-01-15 13:19'
excerpt: What is Eigen Value Decomposition?
comments: true
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
