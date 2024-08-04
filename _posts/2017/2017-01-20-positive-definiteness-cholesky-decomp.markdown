---
layout: post
title: Math - Positive Definiteness and Cholesky Decomposition
date: '2017-01-20 13:19'
subtitle: A matrix is positive definite, if it always sees the glass as half full. But why does the matrix still go to therapy? To break down its issues with Cholesky decomposition. Just Joking.
comments: true
tags:
    - Math
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

### LLT Decomp. Variant - LDL Decomp.

The vanilla method above suffers from numerical instability from the square root operations. For the same A, one can do $A=LDL^T$, where **L is a lower triangle matrix, D is a diagonal matrix with positive diagonal terms**.

To give a feel:

$$
\begin{gather*}
A = LDL^T = 
\begin{bmatrix}
1 & 0 & 0 \\
L_{21} & 1 & 0 \\
L_{31} & L_{32} & 1
\end{bmatrix}

\begin{bmatrix}
D_1 & 0 & 0 \\
0 & D_2 & 0 \\
0 & 0 & D_3
\end{bmatrix}

\begin{bmatrix}
1 & L_{21} & L_{31} \\
0 & 1 & L_{32} \\
0 & 0 & 1
\end{bmatrix}

\end{gather*}
$$

Then we get:

$$
\begin{gather*}
\begin{bmatrix}
D_1 & 0 & 0 \\
L_{21} D_1 & L_{21}^2 D_1 + D_2 & 0 \\
L_{31} D_1 & L_{31} L_{21} D_1 + L_{32} D_2 & L_{31}^2 D_1 + L_{32}^2 D_2 + D_3
\end{bmatrix}
\end{gather*}
$$

Based on $A$, unleashing the GPU power, we can solve:

1. $D_1$
2. $L_{21}$, $L_{31}$, ...
3. $D_2$
4. $L_{32}$ ...
5. $D_3$...

I hate how some websites throw the math right at us. But with the above example, hopefully it's a bit easier:

$$
\begin{gather*}
D_j = A_{j,j} - \sum_{k=1}^{j-1} L_{j,k}^2 D_k
\\
L_{i,j} = \frac{ \left( A_{i,j} - \sum_{k=1}^{j-1} L_{i,k} L_{j,k} D_k \right) }{D_j} \quad \text{for } i > j
\end{gather*}
$$

### LLT Decomp. Variant - Block Cholesky Decomp.

Block Cholesky Decomp. is basically the same as the above, but just operate on block. It is mainly used in GPU. Here's how:

- Choose a small size `rxr` for each block of A. 

$$
\begin{gather*}
A = \begin{pmatrix}
A_{11} & A_{12} & A_{13} & A_{14} \\
A_{21} & A_{22} & A_{23} & A_{24} \\
A_{31} & A_{32} & A_{33} & A_{34} \\
A_{41} & A_{42} & A_{43} & A_{44}
\end{pmatrix}
,
L = \begin{pmatrix}
L_{11} & L_{12} & L_{13} & L_{14} \\
L_{21} & L_{22} & L_{23} & L_{24} \\
L_{31} & L_{32} & L_{33} & L_{34} \\
L_{41} & L_{42} & L_{43} & L_{44}
\end{pmatrix}
\end{gather*}
$$

- Let's start $A_{11}$ in the $L$ matrix:

$$
\begin{gather*}
A = \begin{bmatrix}
A_{11} & B^T \\
B & \hat{A}
\end{bmatrix}
\quad A_{11} \in \mathbb{R}^{r \times r}
,
L = \begin{bmatrix}
L_{11} & 0^T \\
S & \hat{L}
\end{bmatrix}

\end{gather*}
$$

- From $A=LL^T$, we get:

$$

\begin{gather*}
A_{11} = L_{11}L_{11}^T => L_{11} = chol(A_{11}) [1]
\\
S = BL_{11}^{-T} [2]
\\
\hat{L}\hat{L}^T = \hat{A} - SS^T [3]
\end{gather*}
$$

- For (1), We have chosen $r$ to be small enough, so $chol(A_{11})$ is relatively easy
- Then we can witness the GPU power for (2): 

$$
\begin{gather*}
S = \begin{bmatrix}
L_{21} & L_{31} & L_{41}
\end{bmatrix} ^T
\\
=> 
L_{21} = A_{21}L_{11}^{-T}
\\
L_{31} = A_{31}L_{11}^{-T}
\\
L_{41} = A_{41}L_{11}^{-T}
\end{gather*}
$$

- For (3), once we get $S$, can go ahead and calculate $A'=\hat{A}-SS^T$. Again, this can be done by leveraging the almighty GPU power:

$$
\begin{gather*}
A'_{22} = A_{22} - L_{21}L_{21}^T
\\
A'_{23} = A_{23} - L_{21}L_{31}^T
\\
...
\\
A'_{44} = A_{44} - L_{41}L_{41}^T
\end{gather*}
$$

4. Repeat the whole process again with $A'$

## Why Bother With Matrix Decomps?

1. Less storage space for data
2. faster element-wise processing
3. More numerical stability