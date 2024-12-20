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
& \frac{\partial Log(R_1 R_2)}{\partial R_1} = \lim_{\theta \rightarrow 0} \frac{\text{Log}(R_1 \exp(\theta^{\land}) R_2) - \text{Log}(R_1 R_2)}{\theta}
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

## Exercises

### Find $\frac{\partial R^{-1}p}{\partial R}$ using left and right perturbations

Right Perturbation:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial R^{-1}p}{\partial R} = \lim_{\phi \rightarrow 0} \frac{(Rexp(\phi^{\land}))^{-1} p - R^{-1}p}{\phi}

\\
& = \lim_{\phi \rightarrow 0} \frac{exp(\phi^{\land})^{-1} R^{-1} p - R^{-1}p}{\phi}
\\
& = \lim_{\phi \rightarrow 0} \frac{exp(-\phi^{\land}) R^{-1} p - R^{-1}p}{\phi}
\\
& \approx \lim_{\phi \rightarrow 0} \frac{(I - \phi^{\land}) R^{-1} p - R^{-1}p}{\phi}

\\
& = \lim_{\phi \rightarrow 0} \frac{- \phi^{\land} R^{-1} p}{\phi}

\\
& = \lim_{\phi \rightarrow 0} \frac{(R^{-1} p)^{\land} \phi}{\phi}

\\
& = (R^{-1} p)^{\land}
\end{aligned}
\end{gather*}
$$

Left Perturbation:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial R^{-1}p}{\partial R} = \lim_{\phi \rightarrow 0} \frac{(exp(\phi^{\land}) R)^{-1}p - R^{-1}p }{\phi}
\\
& \frac{\partial R^{-1}p}{\partial R} = \lim_{\phi \rightarrow 0} \frac{R^{-1} exp(\phi^{\land})^{-1} p - R^{-1}p}{\phi}

\\
& \frac{\partial R^{-1}p}{\partial R} = \lim_{\phi \rightarrow 0} \frac{R^{-1} exp(-\phi^{\land}) p - R^{-1}p}{\phi}
\\
& \frac{\partial R^{-1}p}{\partial R} \approx \lim_{\phi \rightarrow 0} \frac{R^{-1} (I - \phi^{\land}) p - R^{-1}p}{\phi}

\\
& \frac{\partial R^{-1}p}{\partial R} = \lim_{\phi \rightarrow 0} \frac{-R^{-1} \phi^{\land} p }{\phi}

\\
& \frac{\partial R^{-1}p}{\partial R} = \lim_{\phi \rightarrow 0} \frac{R^{-1} p^{\land} \phi }{\phi}
\\
& = R^{-1} p^{\land}
\end{aligned}
\end{gather*}
$$

### Find $\frac{\partial R_1R_2^{-1}}{\partial R_2}$ using left and right perturbations

When differentiating a rotation matrix w.r.t another rotation matrix, we assume that we want to find the derivative on so(3), so we do this with `Log(R)` to convert this to so(3)

- Right Perturbation:

$$
\begin{gather*}
\begin{aligned}

& \frac{\partial R_1R_2^{-1}}{\partial R_2} = \lim_{\phi \rightarrow 0} \frac{Log(R_1 (R_2 exp(\phi^{\land}))^{-1}) - Log(R_1R_2)^{-1}}{\phi}

\\
& = \lim_{\phi \rightarrow 0} \frac{Log(R_1 exp(\phi^{\land})^{-1} R_2^{-1})- Log(R_1R_2)^{-1}}{\phi}

\\
& = \lim_{\phi \rightarrow 0} \frac{Log(R_1 exp(-\phi^{\land})R_2^{-1})- Log(R_1R_2)^{-1}}{\phi}

\\
& = \lim_{\phi \rightarrow 0} \frac{Log(R_1 R_2^T R_2exp(-\phi^{\land})R_2^{T}) - Log( R_1R_2^{T})}{\phi}
\\
& = \lim_{\phi \rightarrow 0} \frac{Log(R_1 R_2^T exp(-R_2 \phi^{\land})) - Log(R_1R_2^{T})}{\phi}

\\
& = \lim_{\phi \rightarrow 0} \frac{Log(R_1 R_2^T) - J_r^{-1}(R_1 R_2^T)R_2\phi - Log(R_1R_2^{T})}{\phi}

\\
& = J_r^{-1}(R_1 R_2^T)R_2
\end{aligned}
\end{gather*}
$$

Left Perturbation:

$$
\begin{gather*}
\begin{aligned}

& \frac{\partial R_1R_2^{-1}}{\partial R_2} = \lim_{\phi \rightarrow 0} \frac{Log(R_1 (exp(\phi^{\land}))^{-1} R_2) - Log(R_1R_2)^{-1}}{\phi}

\\
& = \lim_{\phi \rightarrow 0} \frac{Log(R_1 R_2^{-1} exp(-\phi^{\land})) - Log(R_1R_2^{-1})}{\phi}
\\
& \approx \lim_{\phi \rightarrow 0} \frac{Log(R_1 R_2^{-1}) -J_r (R_1 R_2^{-1})\phi - Log(R_1R_2^{-1})}{\phi}
\\
& = J_r Log(R_1 R_2^{-1})
\end{aligned}
\end{gather*}
$$
