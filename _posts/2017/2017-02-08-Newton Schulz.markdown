---
layout: post
title: Math - Approximation to SVD
date: 2017-02-07 13:19
subtitle: How to find Null Space? How to implement that using Singular Value Decomposition (SVD)?
comments: true
tags:
  - Math
---

## Introduction

For a square full-rank matrix $G$, the usual SVD is

$$
G = U\Sigma V^\top
$$

and the closest orthogonal matrix to $G$ in Frobenius norm is the **polar factor**

$$
Q = UV^\top.
$$

But computing the SVD can be expensive. Newton-Schulz gives an iterative way to get approximately the same orthogonalized matrix without explicitly doing an SVD.

---

## Goal

We want

$$
Q^\top Q = I
$$

and $Q$ should be close to $G$. The orthogonalized matrix is

$$
Q = G(G^\top G)^{-1/2}
$$

The hard part is computing $(G^\top G)^{-1/2}$. Newton-Schulz approximates this through matrix iterations.

---

## Newton-Schulz iteration for orthogonalization

A common form is

$$
X_{k+1} = \frac{1}{2}X_k(3I - X_k^\top X_k)
$$

Initialize with a scaled version of $G$:

$$
X_0 = \frac{G}{\|G\|_F}
$$

or sometimes

$$
X_0 = \frac{G}{\|G\|_2}.
$$

Then iterate

$$
X_{k+1} = \frac{1}{2}X_k(3I - X_k^\top X_k).
$$

As $k$ increases, $X_k^\top X_k \to I$, so $X_k$ becomes approximately orthogonal.

The final result is

$$
Q \approx X_k.
$$

---

## Numerical example

Let

$$
G =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}.
$$

First compute its Frobenius norm:

$$
\|G\|_F = \sqrt{1^2 + 2^2 + 3^2 + 4^2} = \sqrt{30} \approx 5.477.
$$

So

$$
X_0 =
\frac{1}{5.477}
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\approx
\begin{bmatrix}
0.183 & 0.365 \\
0.548 & 0.730
\end{bmatrix}.
$$

Now apply Newton-Schulz.

### Iteration 1

$$
X_1 = \frac{1}{2}X_0(3I - X_0^\top X_0)
$$

which gives approximately

$$
X_1 =
\begin{bmatrix}
0.173 & 0.356 \\
0.554 & 0.737
\end{bmatrix}.
$$

This is still not very orthogonal:

$$
X_1^\top X_1
\approx
\begin{bmatrix}
0.337 & 0.470 \\
0.470 & 0.670
\end{bmatrix}.
$$

### Iteration 2

$$
X_2 = \frac{1}{2}X_1(3I - X_1^\top X_1)
$$

gives approximately

$$
X_2 =
\begin{bmatrix}
0.117 & 0.408 \\
0.625 & 0.754
\end{bmatrix}.
$$

Now

$$
X_2^\top X_2
\approx
\begin{bmatrix}
0.404 & 0.519 \\
0.519 & 0.735
\end{bmatrix}.
$$

Still not fully orthogonal, but improving.

### After more iterations

After several iterations, Newton-Schulz converges to approximately

$$
Q \approx
\begin{bmatrix}
-0.514 & 0.857 \\
0.857 & 0.514
\end{bmatrix}.
$$

Check orthogonality:

$$
Q^\top Q
\approx
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}.
$$

So the orthogonalized version of

$$
G =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

is approximately

$$
\boxed{
Q =
\begin{bmatrix}
-0.514 & 0.857 \\
0.857 & 0.514
\end{bmatrix}
}
$$

---

## Why Newton-Schulz is useful

The SVD approach gives

$$
G = U\Sigma V^\top,
\qquad
Q = UV^\top.
$$

This is accurate but expensive.

Newton-Schulz only uses matrix multiplications:

$$
X_k^\top X_k,
\qquad
X_k(3I - X_k^\top X_k).
$$

That makes it attractive on GPUs, because matrix multiplication is highly optimized and parallelizable.

So in practice:

$$
\boxed{Q \approx X_K}
$$

where

$$
X_{k+1} = \frac{1}{2}X_k(3I - X_k^\top X_k).
$$

Usually a small number of iterations, such as $5$ to $10$, is enough if $G$ is well-scaled.
