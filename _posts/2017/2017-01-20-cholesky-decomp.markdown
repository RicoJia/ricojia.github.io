---
layout: post
title: Math - Positive Definiteness and  Cholesky Decomposition
date: '2017-01-20 13:19'
excerpt: "A matrix is positive definite, if it always sees the glass as half full. But why does the matrix still go to therapy? To break down its issues with Cholesky decomposition". Just Joking 😉
comments: true
---

## Positive Definiteness

A matrix is Positive definite when for any x:

$$
\begin{gather*}
x^T A x > 0
\end{gather*}
$$

For an `nxn` real symmetric matrix. It's equivalent to say "it's positive definite" when:

1. All eigen values are positive
    - Imagine the case where one eigen value is $\leq 0$. Then, $v^T A v <=0$, which contradicts with the definition of positive definiteness.
2. There exists a real invertible matrix C such that $A=C^{T}C$
    - Using Eigen Value Decomposition: $A = V \Lambda V^{-1} = V \Lambda V^{T}$. Here, $V$ is an orthonormal matrix consists of A's eigen vectors, $V$ consists of its corresponding eigen values.
    - Using 1, since all eigen values are positive, $\Lambda = \sqrt{\Lambda}\sqrt{\Lambda^T}$.
    - So, $A = V \Lambda V^{T} = (\sqrt{\Lambda} V^T)^T (\sqrt{\Lambda} V^T)$

## Cholesky Decomposition (LLT Decomp.)

Definition: if $A$ is a **symmetric** positive definite matrix, there exists a unique lower triangular matrix L such that $A=LL^T$

L looks like:

$$
L = \begin{pmatrix}
l_{11} & 0 & \cdots & 0 \\
l_{21} & l_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
l_{n1} & l_{n2} & \cdots & l_{nn}
\end{pmatrix}
\begin{gather*}

\end{gather*}
$$

Let's focus on 1 row for now:
$$
\begin{gather*}
A = \begin{bmatrix}
a_{11} & A_{21}^T \\
A_{21} & A_{22}
\end{bmatrix}
,
L = \begin{bmatrix}
l_{11} & 0 \\
L_{21} & L_{22}
\end{bmatrix}

,
L^T = \begin{bmatrix}
l_{11} & L_{21}^T \\
0 & L_{22}^T
\end{bmatrix}
\end{gather*}
$$

So $a_{11}$, $l_{11}$ are scalars. Then, we can solve for $l_11$, $l_21$ by:

$$
\begin{gather*}
a_{11} = l_{11}^2
\\
A_{12}^T = l_{11}L_{21}^T
\end{gather*}
$$

But we can't solve for $A_{22}$ directly yet:

$$
\begin{gather*}
A_{22} = L_{21}L_{21}^T + L_{22}L_{22}^T
\end{gather*}
$$

But, isn't this ready for another LLT Decomposition?

$$
\begin{gather*}
A_{22} - L_{21}L_{21}^T
\end{gather*}
$$

Voila, after solving for these iteratively, we get the LLT decomposition $A=LL^T$.

### LLT Decomp. Variant - LDLT Decomp.

The vanilla method above suffers from numerical instability from the square root operations. So, 

TODO

### LLT Decomp. Variant - Block Cholesky Decomp. 

Block Cholesky Decomp. is basically the same as the above, but just operate on block. It is mainly used in GPU. Here's how:

1. Choose a small size `rxr` for each block of A. 

$$
\begin{gather*}
A = \begin{pmatrix}
A_{11} & A_{12} & A_{13} & A_{14} \\
A_{21} & A_{22} & A_{23} & A_{24} \\
A_{31} & A_{32} & A_{33} & A_{34} \\
A_{41} & A_{42} & A_{43} & A_{44}
\end{pmatrix}
\end{gather*}
$$

2. For a single block, say $A_{11}$, in the larger matrix:

$$A = \begin{bmatrix}
A_{11} & B^T \\
B & \hat{A}
\end{bmatrix}
\quad A_{11} \in \mathbb{R}^{r \times r}
\begin{gather*}
\end{gather*}
$$




## Why Bother With Matrix Decomps?

1. Less storage space for data
2. faster element-wise processing
3. More numerical stability