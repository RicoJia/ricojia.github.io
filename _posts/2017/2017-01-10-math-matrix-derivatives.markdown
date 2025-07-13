---
layout: post
title: Math - Matrix Derivatives 
date: '2017-01-10 13:19'
subtitle: Jacobian and Hessian, Vector-Vector Derivative
comments: true
tags:
    - Math
---

## Jacobian and Hessian

### Jacobian

Suppose we have a vector-valued function

$$
\begin{gather*}
\begin{aligned}
& f = [f_0(x), ... f_m(x)]
\\
& x = [x_0, ... x_n]
\end{aligned}
\end{gather*}
$$

Jacobian is its first derivative:

$$
\begin{gather*}
\begin{aligned}
J =
\begin{bmatrix}
\frac{\partial f_0}{\partial x_0} & \dots & \frac{\partial f_0}{\partial x_n} \\
\vdots
\\
\frac{\partial f_m}{\partial x_0} & \dots & \frac{\partial f_m}{\partial x_n}

\end{bmatrix}
\end{aligned}
\end{gather*}
$$

So using first order Taylor Expansion, one can approximate:

$$
\begin{gather*}
\begin{aligned}
& f(x_0 + \Delta x) \approx  f(x_0) + J \Delta x
\end{aligned}
\end{gather*}
$$

### Hessian

Hessian (**for scalar functions only**) is:

$$
\begin{gather*}
\begin{aligned}
& H =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_0 x_0} & \frac{\partial^2 f}{\partial x_0 x_1} & \dots & \frac{\partial^2 f}{\partial x_0x_n}
\\

\vdots
\\

\frac{\partial^2 f}{\partial x_n x_0} & \frac{\partial^2 f}{\partial x_n x_1} & \dots & \frac{\partial^2 f}{\partial x_n x_n}
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

## Vector to Vector Derivatives

An example is rotation. If we rotate a vector:

$$
\begin{gather*}
b = Ra
\\
= \begin{bmatrix}
R_1 & R_2 & R_3
\end{bmatrix}

\begin{bmatrix}
a_1 \\ a_2 \\ a_3
\end{bmatrix}

\\

= \begin{bmatrix}
R_1a_1 + R_2a_2 + R_3a_3
\end{bmatrix}

= \begin{bmatrix}
b_1 \\ b_2 \\ b_3
\end{bmatrix}

\end{gather*}
$$

The jacobian of `b` w.r.t `a` is

$$
\begin{gather*}
\frac{\partial b}{\partial a} =
\begin{bmatrix}
\frac{\partial b_1}{\partial a_1} & \frac{\partial b_1}{\partial a_2} \dots \\
\frac{\partial b_2}{\partial a_1} & \frac{\partial b_2}{\partial a_2} \dots
\end{bmatrix}
\\
= \begin{bmatrix}
\frac{R_1a_1}{a_1} & \frac{R_2a_2}{a_2} & \frac{R_3a_3}{a_3}
\end{bmatrix}

\\
= \begin{bmatrix}
R_1 & R_2 & R_3
\end{bmatrix}
= R
\end{gather*}
$$

### Rules

$$
\begin{gather*}
& \frac{\partial v^T v}{\partial x} = \frac{\partial v^T}{\partial x} v + \frac{\partial v^T}{\partial x} v^T \tag{1}
\\
& \frac{\partial v v^T}{\partial x} = \frac{\partial v}{\partial x} v^T + v \frac{\partial v^T}{\partial x} \tag{2}
\\
& \frac{\partial v}{\partial x^T} = (\frac{\partial v^T}{\partial x})^T \tag{3}
\end{gather*}
$$

### Examples

$$
\begin{gather*}
\begin{aligned}
& \text{using (2): } \frac{\partial x^T A x}{\partial x^T} = (A + A^T)x
\end{aligned}
\end{gather*}
$$

## TODO

Notes: <https://github.com/RicoJia/notes/wiki/Math#basic-calculus>

<https://cs231n.stanford.edu/vecDerivs.pdf>
