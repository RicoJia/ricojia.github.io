---
layout: post
title: Math - Eigen Value, Eigen Vector, and Eigen Value Decomposition
date: '2017-02-07 13:19'
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

So, 
