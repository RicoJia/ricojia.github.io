---
layout: post
title: Math - Lagrange Multiplier
subtitle: Pontryagin's Minimum Principle
date: '2017-02-04 13:19'
header-img: "img/bg-material.jpg"
tags:
    - Math
---

## Lagrange Multiplier

Multivating example: maximize $f(x,y)$, where $xy+1$, given constraint $g(x,y) = x^2+y^2-1 = 0$

Geometric Intuition: the value of ```f(x,y)``` and the constraint **it must stay on** are tangent to each other. That is, a small perturbation along the constraint curve will not cause change in the value function, hence a potential extrema is achieved.

<p align="center">
    <img src="https://user-images.githubusercontent.com/39393023/122949585-670a5f00-d341-11eb-8cd1-9055c7238239.png" height="300" width="width"/>
</p>

So this is equivalent to: $L=f(x)-\lambda g(x)$, and get $[\frac{\partial{L}}{\partial{x}}, \frac{\partial{L}}{\partial{\lambda}}] = 0$

$\lambda$ is **the lagrange multiplier**.

To solve: 

1. Define Lagrangian $L = f(x) + \sum_i \lambda_i g_i(x,y)$. In this case, it's simply:

$$
\begin{gather*}
\begin{aligned}
& L = f(x) + g(x,y) = xy + 1 + \lambda(x^2+y^2-1)
\end{aligned}
\end{gather*}
$$

2. Now, get the first order derivatives: 

$$
\begin{gather*}
\begin{aligned}
& \nabla L = [\frac{\partial L}{\partial x}, \frac{\partial L}{\partial y}, \frac{\partial L}{\partial \lambda}]
\\ & = [y - 2 \lambda x, x - 2\lambda y, x^2 + y^2 - 1]
\end{aligned}
\end{gather*}
$$

3. We now let the derivates to be 0 (note we are explicitly adding constraint here):

$$
\begin{gather*}
\begin{aligned}
& y = 2 \lambda x
\\ &
x = 2 \lambda y
\\ &
x^2 + y^2 - 1 = 0
\end{aligned}
\end{gather*}
$$

We can get the solution: $\lambda = \frac{1}{2}, x = y = \sqrt{\frac{1}{2}}$

### Why does Lagrange Multiplier work?

Since the optimal point is on multiple constraint surfaces, the pertabtion on each surface must be perpendicular to its surface normal: $\nabla g_k(x, y)$:

$$
\begin{gather*}
\begin{aligned}
& \nabla g_k(x, y) = [\frac{\partial g(x, y)}{\partial x}, \frac{\partial g(x, y)}{\partial y}]
\\ &
d^T \nabla g_k(x, y) = 0
\end{aligned}
\end{gather*}
$$

At the optimal point, any disturbance `d` would also satisfy:

$$
\begin{gather*}
\begin{aligned}
& d^T \nabla f(x,y) = 0
\end{aligned}
\end{gather*}
$$

So the surface normals and the surface normal of the surface under optimization are parallel:

$$
\begin{gather*}
\begin{aligned}
& \nabla f(x,y) = \lambda_1 \nabla g_1(x, y)
\\ & 
...
\\ & 

\nabla f(x,y) = \lambda_k \nabla g_k(x, y)
\\ &
\Rightarrow \nabla f(x,y) + \sum_k \lambda_k g_k(x, y) = 0 
\Rightarrow \nabla L = 0
\end{aligned}
\end{gather*}
$$

## Pontryagin's Minimum Principle

If f(x,y, t) is a function of time, we will have "costates" $\lambda_k(t)$. So:

If we define our state transition to be: 
$$
\begin{gather*}
\begin{aligned}
& x'(t) = f(x(t), u(t)), x(0) = u(0)
\end{aligned}
\end{gather*}
$$

And we want to minimize totla control effort `u`:

$$
\begin{gather*}
\begin{aligned}
& J = \int_0^T L(x(t), u(t)) dt + h(x(T))
\end{aligned}
\end{gather*}
$$

Pontryagin’s Minimum Principle says any optimal pair `x*(.), u*(.)` must have a costate $\lambda(t)$ satisfying the hamiltonian:

$$
\begin{gather*}
\begin{aligned}
& H(x, u, \lambda) = L(x,u) + \lambda^T f(x,u)
\end{aligned}
\end{gather*}
$$

Then State Dynamics:

$$
\begin{gather*}
\begin{aligned}
& \dot x^*(t) = \frac{\partial H}{\partial \lambda} (x^*(t), u^*(t), \lambda(t)), x^*(0) = x_0
\end{aligned}
\end{gather*}
$$

$\lambda(t)$: 

$$
\begin{gather*}
\begin{aligned}
& \lambda(t) = -\frac{\partial H}{\partial x} (x^*(t), u^*(t), \lambda(t))
\end{aligned}
\end{gather*}
$$

With the boundary condition:

$$
\begin{gather*}
\begin{aligned}
& \lambda(T) = \frac{\partial H}{\partial x}(x^*(T))
\end{aligned}
\end{gather*}
$$

Optimal Control Output is:

$$
\begin{gather*}
\begin{aligned}
& u^*(t) = argmin H(x^*, u, \lambda)
\end{aligned}
\end{gather*}
$$