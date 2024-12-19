---
layout: post
title: Robotics Fundamentals - Velocities
date: '2024-03-15 13:19'
subtitle: Velocities, Accelerations, Derivative of Rotations
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

We know that `SO(3)` is a manifold that do not support direct addition. So, we need to come back to the very definition of derivatives - we perturb `Ra` in terms of the rotation vector $\theta$ using either the left/right perturbation model, then because addition is supported among rotation vectors, we calculate the derivative there. Here we use the right perturbation model since it's more common:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial Ra}{\partial R} =
\lim_{\theta \rightarrow 0} \frac{R exp(\theta^{\land}) a}{\theta}

\\
& \approx \lim_{\theta \rightarrow 0} \frac{R (I + \theta^{\land}) a}{\theta} = \lim_{\theta \rightarrow 0} \frac{ 0 + R (\theta^{\land}) a}{\theta}

= \lim_{\theta \rightarrow 0} \frac{ R (-a^{\land}) \theta}{\theta}
\\
& = -Ra^{\land}
\end{aligned}
\end{gather*}
$$

### Derivative of Rotations Is The Same For Quaternions and Rotation Matrix

Recall that rotation in quaternion is $p'=qaq*$. If `q=[s, v]`, withthout proof,

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial (qaq*)}{\partial q} = 2[wa + v^{\land} a, v^{T}aI_3 + va^T - av^T - wa^{\land}]
\end{aligned}
\end{gather*}
$$

To get the derivative w.r.t quaternion `q`, one can use the right perturbation model as well. If we perturb `Ra` by `[0, w]`, that's equivalent to adding `[1, 0.5w]` to the quaternion. To be consistent with the `SO(3)` representation, we should be getting the same value:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial Ra}{\partial w} = -Ra^{\land}
\end{aligned}
\end{gather*}
$$

This tells us that when updating rotation parameters, we use the same Jacobian for both the quaternions and the rotation matrices. (TODO: Not sure why?)

## Derivative of `SO(3)` to `so(3)`

To recover a composite rotation: `Log(R1R2)` to `so(3)`, we can find its derivative w.r.t `R1` by applying the right perturbation:

$$
\begin{gather*}
& \frac{\partial R_1 R_2}{\partial R_1} = \lim_{\theta \rightarrow 0} \frac{\text{Log}(R_1 \exp(\theta^{\land}) R_2) - \text{Log}(R_1 R_2)}{\theta}
\\
& = \lim_{\theta \rightarrow 0} \frac{\text{Log}(R_1 R_2 R_2^T \exp(\theta^{\land}) R_2) - \text{Log}(R_1 R_2)}{\theta}
\\
& = \lim_{\theta \rightarrow 0} \frac{\text{Log}(R_1 R_2  \exp((R_2^T\theta)^{\land}) ) - \text{Log}(R_1 R_2)}{\theta}
\tag{1}
\\
& = \lim_{\theta \rightarrow 0} \frac{\text{Log}(R_1 R_2) + J_r^{-1}\text{Log}(R_1 R_2)\text{Log}(\exp((R_2^T\theta)^{\land})) - \text{Log}(R_1 R_2)}{\theta}
\tag{2}
\\
& = J_r^{-1}\text{Log}(R_1 R_2)R_2^T    \tag{3}
\end{gather*}
$$

(1) is using the property with proof:

$$
\begin{gather*}
\begin{aligned}
& R^T exp(\theta^{\land}) R = exp((R^T \theta)^{\land})
\end{aligned}
\end{gather*}
$$

- To prove (1), first check out the section **Rotation Preserves Dot Product** for proving $R^T \theta^{\land} R = (R^T \theta)^{\land}$. Then, since

$$
\begin{gather*}
\begin{aligned}
& exp(\theta^{\land}) = I + \theta^{\land} + \frac{(\theta^{\land})^2}{2!} + ...
\\
& \rightarrow R^T exp(\theta^{\land}) R = R^T (I + \theta^{\land} + \frac{(\theta^{\land})^2}{2!} + ...) R = R^T R + R^T \theta^{\land} R + \frac{(R^T \theta^{\land} R)^2}{2!} + ...
\\
& = I + (R^T \theta)^{\land} + \frac{((R^T \theta)^{\land})^2}{2!} + ...
\\
& = exp((R^T \theta)^{\land})
\end{aligned}
\end{gather*}
$$

- (2) is a first order BCH approximation:

$$
\begin{gather*}
\begin{aligned}
& Log(AB)\approx J_r^{-1} Log(A)Log(B) + Log(A)
\end{aligned}
\end{gather*}
$$

Similarly,

$$
\begin{gather*}
& \frac{\partial R_1 R_2}{\partial R_2} = J_r^{-1}Log(R_1 R_2) \tag{4}
\end{gather*}
$$

**The above equations (3) and (4) are VERY INPORTANT for LiDAR SLAM!!**

### Rotation Preserves Dot Product

- Prove $R^T \theta^{\land} R = (R^T \theta)^{\land}$:

$$
\begin{gather*}
\begin{aligned}
& R^T \theta^{\land} Rv = R^T (\theta \times (Rv))
\\
& (R^T \theta)^{\land} v = (R^T \theta) \times (v)
\\
& R^T (\theta \times (Rv)) = (R^T \theta) \times (R^TRv) = (R^T \theta) \times (v)
\end{aligned}
\end{gather*}
$$

- This is because "Rotation Preserves Dot Product". Why? Because dot product is the unique vector $Ra \times Rb = |Ra||Rb|sin\theta \rightarrow (Rn) = R(a \times b)$
