---
layout: post
title: Robotics Fundamentals - Velocities
date: '2024-03-15 13:19'
subtitle: Velocities, Accelerations
comments: true
tags:
    - Robotics
---

## Rotation-Only Velocity and Acceleration

Assume we have a world frame, a car frame, and a point. The car is rotating around the world frame, no translation.The point itself is moving as well, $p_c'$.

So in the world frame, the velocity of the point $p_w$ can be determined from its position and velocities in the car frame

$$
\begin{gather*}
p_w = R_{wc} p_c
\\
(p_w)' = R_{wc}' p_c + R_{wc} p_c'
\\ = R_{wc}(w^{\land} p_c + p_c')
\end{gather*}
$$

**Note that this is NOT converting the velocity vector $p_c$.** This is instead finding the velocity of the point given the rotation of the car, and the car's perceived point velocity $p_c$.

The acceleration would be:

$$
\begin{gather*}
p_w'' = R_{wc}'(w^{\land} p_c + p_c') + R_{wc}(w^{\land} p_c + p_c')'
\\
= R_{wc}w^{\land}(w^{\land} p_c + p_c') + R_{wc}((w^{\land})' p_c + w^{\land} p_c' + p_c'')
\\
= R_{wc}(w^{\land}w^{\land} p_c + (w^{\land})' p_c + 2w^{\land} p_c' + p_c'')
\end{gather*}
$$

Usually, in non-high-accuracy scenarios, we only keep `p_c''` and ignore the linear accelerations caused by rotations.

$$
\begin{gather*}
p_w'' = R_{wc} p_c''
\end{gather*}
$$

## Derivative of Rotations, $\frac{\partial Ra}{\partial R}$

If we rotate a vector a, what's its derivative w.r.t R? That is, when there's an infinitesmal change in R, what would be that change in a?

We know that `SO(3)` is a manifold that do not support direct addition. So, we need to come back to the very definition of derivatives - we perturb `Ra` using either the left/right perturbation model, then measure the derivative:

$$
\begin{gather*}
\frac{\partial Ra}{\partial R} =
\lim_{\theta \rightarrow 0} \frac{R exp(\theta^{\land}) a}{\theta}

\\
\approx \lim_{\theta \rightarrow 0} \frac{R (I + \theta^{\land}) a}{exp(\theta^{\land})}
\end{gather*}
$$
