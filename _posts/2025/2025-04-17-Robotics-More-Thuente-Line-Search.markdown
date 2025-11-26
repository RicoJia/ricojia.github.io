---
layout: post
title: Robotics - Moré-Thuente Line Search Algorithmk
date: '2025-04-17 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---


## Introduction

Introduced in the early 1990s, line search and trust-region strategies have become standard tools for improving the robustness of iterative optimization methods used in LiDAR odometry—such as NDT, ICP, and their many variants.

In these methods, we iteratively refine a pose vector 𝑥 by solving an optimization problem that minimizes a cost function f(x). At each iteration, we compute a search direction dx using an algorithm such as Gauss–Newton or Levenberg–Marquardt.

This search direction tells us **which way to move, but not how far**. So instead of taking the full step x = 𝑥 \oplus  𝑑𝑥, which may overshoot the minimum or even increase the cost, we introduce a step size (or scale) 𝛼.

$$
x = x + \alpha\,dx
$$

The goal is to choose 𝛼 ∈ ( 0 , 1 ] such that the updated pose reduces the cost function more reliably than taking the raw step dx. This process is known as line search.

## Problem Setup

Given the current pose $x_0$​, the cost w.r.t $\alpha$ along the search direction $dx$ is:

$$\phi(\alpha) = f\bigl(x_0 + \alpha\,dx\bigr).$$

We want: 

$$
f\bigl(x_0 + \alpha\,dx\bigr) \le f(x_0 + dx)
$$

## Armijo Condition

We always start from $\alpha=1$. The first intuition is that if our step is not too large, the cost should be smaller than the first-order Taylor approximation around $x_0$. In math form we use the Armijo (sufficient-decrease) idea:

$$f\bigl(x_0 + \alpha\,dx\bigr) \le f(x_0) + c_1\,\alpha\,\nabla f(x_0)^\top dx,$$

where $c_1$ is a small constant (commonly $c_1=10^{-4}$). (This is a relaxed version of the first-order Taylor expansion)

The directional derivative of $\phi$ is

$$\frac{d}{d\alpha}f\bigl(x_0 + \alpha\,dx\bigr) = \nabla f\bigl(x_0 + \alpha\,dx\bigr)^\top dx,$$

Evaluated at $\alpha=0$,

$$
\phi'(0)=\nabla f(x_0)^\top dx
$$


### When Armijo is violated, what went wrong?

When the Armijo sufficient-decrease condition fails, there are two possible causes:

1. The step is too large (overshoot).

    - The update x 0 +αdx jumps past the minimum and the cost goes back up. In this case, reducing the step size helps:
    - The update $x_0+\alpha\,dx$ jumps past the minimum and the cost increases. In this case, reducing the step size helps, e.g.

$$
\alpha \leftarrow 0.5\alpha
$$ 

2. The step is too small. The decrease in the cost is smaller than the linear model predicts, and Armijo rejects the step even though it is not too large. This can happen when:
    - The search direction dx is nearly orthogonal to the gradient,
    - The iterate is near a saddle point,
    - Or the gradient is noisy.

In this case, do we do $\alpha = max(2\alpha, \alpha_bracket)$

How do we tell whether α is too large or too small?

We check the directional derivative at the candidate step:

$$
\phi'(\alpha)=\nabla f\bigl(x_0+\alpha\,dx\bigr)^\top dx
$$ 

- If $\phi'(\alpha)<0$: the slope is still negative → the cost is still decreasing → $\alpha$ is too small (we haven't reached the bottom yet).
- If $\phi'(\alpha)>0$: the slope became positive → we passed the minimum → $\alpha$ is too large.

## Strong Curvature Check (strong Wolfe curvature condition)

After Armijo is satisfied, we check for the stopping condition: whether the directional derivative

ϕ'(α) is sufficiently close to zero.

If 

$$
|\phi'(\alpha)|\le c_2\,|\phi'(0)|, c_2 \in (c_1, 1)
$$ 

we say the slope is flat enough and accept $\alpha$. This is the (strong) Wolfe curvature condition. 

