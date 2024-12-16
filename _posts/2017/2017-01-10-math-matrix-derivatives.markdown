---
layout: post
title: Math - Matrix Derivatives 
date: '2017-01-20 13:19'
subtitle: Foundation For Everything in Deep Learning and 3D SLAM
comments: true
tags:
    - Math
---

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

## TODO

Notes: <https://github.com/RicoJia/notes/wiki/Math#basic-calculus>

<https://cs231n.stanford.edu/vecDerivs.pdf>
