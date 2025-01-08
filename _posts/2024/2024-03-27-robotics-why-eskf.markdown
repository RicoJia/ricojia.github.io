---
layout: post
title: Robotics - [ESKF Series 5] Why Using Error-State Kalman Filter (ESKF) For IMU
date: '2024-03-24 13:19'
subtitle: ESKF, GINS
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## ESKF Handles Rotations on SO(3) Better

Key issue: Rotations lie on the special orthogonal group SO(3), which is a nonlinear manifold (not a simple vector space like position). A standard EKF typically performs state updates in a “vector space” manner: it linearizes around the current state and then does `state←state+Δx`. But for 3D rotations, you can’t simply “add” the rotation increment `Δθ` directly to the existing orientation representation if you want to stay on SO(3).

- Vanilla EKF approach:
    - If you use a quaternion or rotation matrix as your rotation representation, then you have to define the addition of Δx. A naive approach might be:

    ```
    q_new=q_old+Δθ
    ```
    
    - But quaternion addition is not the correct way to update orientation. You would need to an exponential map that properly keeps you on the manifold.

- ESKF approach
    - You store the nominal orientation (often as a quaternion or rotation matrix) as a separate entity, then maintain an error-state `δθ` in a minimal 3D representation (e.g., in the tangent space).
    - When you do an EKF update, you update only this small error δθ, and then “correct” the nominal orientation using an exponential map:

    ```
    R_new=exp⁡ ⁣(δθ∧) R_old (where ∧ maps a 3D vector to a skew-symmetric matrix for rotations).
    ```
    - By doing this, you ensure the orientation remains on SO(3) after an update, and the linearization is better-conditioned because you are linearizing around a small error state rather than the full orientation.

### Could we define a “generic addition” for EKF?

In theory, yes. Defining a proper “retraction” (or group operation) for the rotation part of the state is essentially how manifold-based filters are derived. For example, we do the final $x_{t+1} = x_t \oplus \Delta x_t$. In that case, our EKF motion model is actually following:

$$
\begin{gather*}
\begin{aligned}
& x_t{t+1}^* = f(x_t, u_t)
\\
& x_{t+1} \approx x_{t+1}^* + F_t(x_{t+1} \ominus x_t) |_{x = x_t}
\end{aligned}
\end{gather*}
$$

- $F_t$ is the Jacobian of the motion model and is linearzed $x_t$. $x_t$ could be relatively large.
    - If you are using Euler Angles, when the rotation angle is large, there could be singularities around gimbal lock values. E.g, `pitch = ±90∘`
    - Even if you are not using Euler Angles and are using quaternion or world frame rotation vector instead, at certain large values, the first order Jacobian may not model the system well using linearization. **So, linearization around $\delta x$ in ESKF is around 0 and generally have better accuracies**


- Isn't $x_{t+1} \ominus x_t$ ESKF?


## ESKF Can Work With Higher Floating Point Precision

A typical `float32` has around 7 significant digits of precision, a `double` has 15-16 decimal digits of precision. In [UTM coordinates](./2024-03-23-robotics-gps-utm.markdown), one needs at least [8 significant digits to represent centimeter level accuracy.](./2024-03-23-robotics-gps-utm.markdown) 

However, kalman filter update is rather small. Typically, RTK GPS operates at 1hz. If we have a vehicle travelling at 10km/h in congested areas, that's roughly 2.8m/s. Here in low speed scenarios, centimeter level accuracy will give us a lot of leg room. If we work with float32 in this case, centimeter updates will be rounded off when adding it to `10^6m` coordinates.

- In EKF, states are GPS coordinates. We linearize around the last state, propagating the large coodinates throught the covariance matrix, and add a small correction to it to update:

$$
\begin{gather*}
\begin{aligned}
& H = \frac{\partial h(x)}{\partial x}|_{x=x_{t+1}^*} = \begin{bmatrix}
\frac{x - c_x}{d} & \frac{y - c_y}{d} & 0 \\
\frac{c_y - y}{d^2} & \frac{x - c_x}{d^2} & -1
\end{bmatrix}

\\ 
& S = H P_{t+1}^{*} H^\top + R
\\ & 
K = P_{t+1}^{*} H^\top S^{-1}
\end{aligned}
\end{gather*}
$$

Along this process we operate on `10^6m` position state variables. Any associated operations, such as motion model prediction, would not be able to achieve centimeter accuracy. So **throughout EKF, we need `FP64`**

- On the other hand, in ESKF, we operate on the error $\delta x$, which stays close to 0. All the intermediate steps just require `FP32` to achieve centimeter level accuracy. Of course, at the end, we still need to add the small update back to the large coordinates in `FP64`, but it's fairly minimal.

### Consequently, It Is Imporant To...

- It is important to use `FP64` for ESKF update
- Run the filter at a high enough frequency so that each discrete step is small
- Be careful with angle wraps for $[0, 2 \pi]$