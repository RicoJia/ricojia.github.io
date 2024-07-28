---
layout: post
title: Math - Gram Schmidt Orthogonolization And QR Decomposition
date: '2017-01-05 13:19'
excerpt: Super useful in finding forming an orthogonal vector basis, e.g., Singular Value Decomposition
comments: true
---

## Background Knowledge

### Glossary

- Inner product: $<a,b> = \vec{a}^T \cdot \vec{b}$, which is a.k.a "dot product"
- Outer product: $a \otimes b = \vec{a} \cdot \vec{b}^T$, which results in a matrix.

### Vector Projection

Projection of a on to b $proj_ba= \frac{<a,b>}{\sqrt{<a,a>}}a$

<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/c646e1e5-2e46-4348-a862-be1dc63fd3f0" height="200" alt=""/>
        <figcaption><a href="https://www.google.com/url?sa=i&url=https%3A%2F%2Flearninglab.rmit.edu.au%2Fcontent%2Fv5-projection-vectors.html&psig=AOvVaw3fOwPqOMCslWyRPBmD2gXE&ust=1722182169016000&source=images&cd=vfe&opi=89978449&ved=0CBQQjhxqFwoTCPjUodXKx4cDFQAAAAAdAAAAABAJ">Source: RMIT Learning Lab</a></figcaption>
    </figure>
</p>

## Gram-Schmidt Orthogonalization

**Goal**: given m linearly independent N-dimensional vectors ${v_1 ... v_m}$, transform them into an orthogonal set of vectors.

**Method**: an orthogonal vector is formed by subtracting a vector's projection onto other orthogonal basis, $U$.
1. $u_1 = v_1$
2. $u_2 = v_2 - \frac{<v_2,u_1>}{\sqrt{<u_1,u_1>}}u_1$
3. $u_3 = v_3 - \frac{<v_3,u_1>}{\sqrt{<u_1,u_1>}}u_1 - \frac{<v_3,u_2>}{\sqrt{<u_2,u_2>}}u_2$

**Caveats**

- It's not numerically stable - rounding error in each step can be accumulated. In real life, people use Householder transformation, or Givens rotation.

## QR Decomposition

**Goal**: given a matrix $A$, decompose it into $A=QR$ such that Q is orthonormal, and R is an upper triangular matrix. 

**How**:
1. Get an Orthogonal Basis of $A$'s columns using Gram-Schmidt (or Householder Transformation). This basis is $Q$
2. Get $R$:

$$
\begin{gather*}
A = QR 
\\ => Q^T A = Q^T QR 
\\ => R = Q^T A
\end{gather*}
$$

Example:

$$
\begin{gather*}
A = \begin{bmatrix}
1 & 1 & 0 \\
1 & 0 & 1 \\
0 & 1 & 1
\end{bmatrix}
\end{gather*}
$$

The gram schmidt process:

$$
\begin{equation*}
\begin{aligned}

\mathbf{u}_1 = \mathbf{a}_1 = (1, 1, 0),

\\

\mathbf{e}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|} = \frac{1}{\sqrt{2}} (1, 1, 0) = \left( \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0 \right),

\\

\mathbf{u}_2 = \mathbf{a}_2 - (\mathbf{a}_2 \cdot \mathbf{e}_1)\mathbf{e}_1 = (1, 0, 1) - \frac{1}{\sqrt{2}} \left(1, 1, 0\right) = \left(1 - \frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}, 1\right),

\\
\mathbf{e}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|} = \frac{1}{\sqrt{3/2}} \left( \frac{1}{2}, -\frac{1}{2}, 1 \right) = \left( \frac{1}{\sqrt{6}}, -\frac{1}{\sqrt{6}}, \frac{2}{\sqrt{6}} \right),

\\
\mathbf{u}_3 = \mathbf{a}_3 - (\mathbf{a}_3 \cdot \mathbf{e}_1)\mathbf{e}_1 - (\mathbf{a}_3 \cdot \mathbf{e}_2)\mathbf{e}_2

\\
= (0, 1, 1) - \frac{1}{\sqrt{2}} \left(1, 1, 0\right) - \frac{1}{\sqrt{6}} \left(1, -1, 2\right)

\\
= \left(0 - \frac{1}{\sqrt{2}} - \frac{1}{\sqrt{6}}, 1 - \frac{1}{\sqrt{2}} + \frac{1}{\sqrt{6}}, 1 - \frac{2}{\sqrt{6}}\right) = \left(-\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}\right),

\\
\mathbf{e}_3 = \frac{\mathbf{u}_3}{\|\mathbf{u}_3\|} = \left( -\frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}}, \frac{1}{\sqrt{3}} \right).

\end{aligned}
\end{equation*}
$$

One important note from Gram-Schmidt, is that each subsequent basis vector is perpendicular to the vectors that creates the previous basis vectors, but not to the vector that it directly comes from. E.g., $e_2\perp a_1$, $e_3\perp a_1$, $e_3\perp a_2$, $e_2 \not\perp a_2$

Then, Q is 

$$
\begin{gather*}
Q = \begin{bmatrix}
e_1 & e_2 & e_3
\end{bmatrix}
= 
\begin{bmatrix} 
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} & -\frac{1}{\sqrt{3}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{6}} & \frac{1}{\sqrt{3}} \\
0 & \frac{2}{\sqrt{6}} & \frac{1}{\sqrt{3}}
\end{bmatrix}
\end{gather*}
$$

$$
\begin{gather*}
R = Q^T A = \begin{bmatrix} 
\mathbf{a}_1 \cdot \mathbf{e}_1 & \mathbf{a}_2 \cdot \mathbf{e}_1 & \mathbf{a}_3 \cdot \mathbf{e}_1 \\
0 & \mathbf{a}_2 \cdot \mathbf{e}_2 & \mathbf{a}_3 \cdot \mathbf{e}_2 \\
0 & 0 & \mathbf{a}_3 \cdot \mathbf{e}_3 
\end{bmatrix} 
= 
\begin{bmatrix} 
\frac{2}{\sqrt{2}} & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
0 & \frac{2}{\sqrt{6}} & \frac{1}{\sqrt{6}} \\
0 & 0 & \frac{2}{\sqrt{3}}
\end{bmatrix}
\end{gather*}
$$

When A is `m x n`, Q will be `m x n` and R will be `n x n`. If A has full rank, then Q will have full rank, so R's diagnoal terms will be non-zero. Then R is invertible. 

On the other hand, if A doesn't have full rank, some of its column vectors are linearly dependent on others. Their corresponding diagonal terms in R, will be zero, because the orthonormal vectors will be zero vectors (because they derive from previous orthonormal basis). In that case, R will not be invertible.

When `m < n`, A is overdetermined. So it will defintely have linearly dependent columns. Q will be `mxm`, and R will be `mxn`. In that case, R will not be a square matrix.

### Applications

1. Find Least Square Solution to $Ax = b$. The common solution is 

$$
\begin{gather*}
\hat{x} = (A^TA)^{-1}A^Tb
\end{gather*}
$$

Using QR decomposition:

$$
\begin{gather*}
\hat{x} = (A^TA)^{-1}A^Tb
\\ = (R^TR)^{-1}R^TQ^T b
\end{gather*}
$$

Note that $R$ may not be square matrix (when `m<n`), or may not be invertible (when A is not invertible). Except from those cases, 

$$
\begin{gather*}
\hat{x} = R^{-1}Q^T b
\\
R\hat{x} = Q^T b
\end{gather*}
$$

One trick for solving for x is 

$$
\begin{gather*}
\begin{bmatrix}
r_{11} & r_{12} & r_{13} \\
0 & r_{22} & r_{23} \\
0 & 0 & r_{33}
\end{bmatrix} \hat{x} = 
\begin{bmatrix}
y_1 \\ y_2 \\ y_3
\end{bmatrix}
\\ => 
y_3 = r_33 
\\...
\end{gather*}
$$

So x can be solved quite easily through substitution.
