---
layout: post
title: Math - Interpoloation
date: '2017-01-26 13:19'
subtitle: Linear, Cubic, Bicubic Splines, Slerp
comments: true
tags:
    - Math
---

## Interpolation

"Interpolation" is essentially "how to get through lines and predict the value of a point in between?". The first step is to connect points. For that, we need **a spline**. A spline is a function defined piecewise by polynomials.

### 1D Case

In 1D, You can simply connect points with straight lines (linear interpolation), or you try to draw a quadratic curve, or even with higher order curves.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/9937a26f-435f-4178-9a3b-8e0c3bb122a0" height="300" alt=""/>
        <figcaption><a href="https://www.mssc.mu.edu/~daniel/pubs/RoweTalkMSCS_BiCubic.pdf">Source</a></figcaption>
    </figure>
</p>
</div>

The second step is to calculate the interpolation value **with a kernel matrix**. Take 1D linear interpolation for example, if we want to get the interpolation value between 2 points, we have

$$
k = \begin{bmatrix}
0.5 & 0.5
\end{bmatrix}
\\
P = kx
$$

In 1D, if we choose to draw a cubic curve, the first step is we need 4 points, and define the spline: $f(x) = ax^3 + bx^2 + cx + d$

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/198ee503-d624-43ec-ad1a-9d763bf43974" height="300" alt=""/>
        <figcaption><a href="https://www.mssc.mu.edu/~daniel/pubs/RoweTalkMSCS_BiCubic.pdf">Source</a></figcaption>
    </figure>
</p>
</div>

Now, if we want to estimate the value at `x=0.5`, we need to solve for `[a, b, c, d]`. We can choose `x=[-1, 0, 1, 2]`, and get

$$
\begin{bmatrix}
f(-1) \\ f(0) \\ f(1) \\f(2)
\end{bmatrix}

=

\begin{bmatrix}
-1 & 1 & -1 & 1 \\
0 & 0 & 0 & 1 \\
1 & 1 & 1 & 1 \\
8 & 4 & 2 & 1
\end{bmatrix}

\begin{bmatrix}
a \\ b \\ c \\d
\end{bmatrix}
$$

Then,

$$

\begin{bmatrix}
d \\ c \\b \\a
\end{bmatrix}

=

\frac{1}{6} \begin{bmatrix}
0 & 6 & 0 & 0 \\
-2 & -3 & 6 & -1 \\
3 & -6 & 3 & 0 \\
-1 & 3 & -3 & 1
\end{bmatrix}

\begin{bmatrix}
f(-1) \\ f(0) \\ f(1) \\f(2)
\end{bmatrix}
$$

Finally, plug `x=0.5, y=0.5` into

$$
estimate = 
\begin{bmatrix}
a \\ b \\ c \\d
\end{bmatrix}^T

\begin{bmatrix}
x^3 \\ x^2 \\ x \\ 1
\end{bmatrix}^T

$$

voila!

**Note on differentiability:** Cubic splines are very common. **they are typically required to be twice differentiable**, so the first and second derivatives are continuous. Linear interpolations (splines) can't enforce differentiablity at connecting points. Higher order splines, like B-splines, Bezier splines, etc., can have higher order derivatives.

### 2D Case

In 2D interpolations, we will similarly work with 2D kernels.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/0d0c13f7-22c5-4555-888a-90f1bfa941da" height="300" alt=""/>
        <figcaption><a href="https://www.mssc.mu.edu/~daniel/pubs/RoweTalkMSCS_BiCubic.pdf">Source</a></figcaption>
    </figure>
</p>
</div>

If we plot the kernel functions:

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/ba9a433d-adb6-4b0d-8c5f-9b72ee5ff607" height="300" alt=""/>
        <figcaption><a href="https://www.mssc.mu.edu/~daniel/pubs/RoweTalkMSCS_BiCubic.pdf">Source</a></figcaption>
    </figure>
</p>
</div>

## Slerp (Spherical Linear Interpolation)

A geodesic is the **locally shortest path between two points** constrained to lie on a surface (or more generally, a manifold).

Slerp interpolates smoothly between two unit quaternions q0 q1 , tracing the geodesic on the 4D unit sphere (valid quaternions that represent rotations) at constant angular velocity.


$$
\theta = \arccos \bigl( \langle q_0, q_1 \rangle \bigr),
$$

The angle between them on $(S^3)$. Then

$$
q(t) =
\frac{\sin\bigl((1-t)\theta\bigr)}{\sin \theta}\, q_0 \;+\;
\frac{\sin(t\theta)}{\sin \theta}\, q_1,
\quad t \in [0,1].
$$

If you think of $q_0, q_1$ as rotation matrices $(R_0, R_1 \in SO(3))$,
then slerp is equivalent to
$$
R(t) = R_0 \exp\!\Bigl( t \,\log\!\bigl(R_0^\top R_1\bigr) \Bigr).
$$
