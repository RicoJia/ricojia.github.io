---
layout: post
title: Robotics - [ESKF Series 4] Full Error-State Kalman Filter (ESKF) in GNSS-Inertial Navigation System (GINS)
date: '2024-03-24 13:19'
subtitle: ESKF, GINS
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

If you haven't checked out a motivational 2D robot example of ESKF, [please check here](./2024-03-25-robotics-full-eskf.markdown).

In this post, $\oplus$ is the "generic add", which "adds" on a manifold.

## ESKF (On Manifold) in GINS (GPS-Intertial Navigation System)

GINS = GNSS + IMU

### [Step 1] States and Motion Model Setup

In a GINS system, we have below states: `[position, velocity, rotation matrix, bias a, bias gyro (rotation), gravity]`
Our state space is: $x = [p_x, p_y, p_z, v_x, v_y, v_z, \theta_x, \theta_y, \theta_z, b_{gx}, b_{gy}, b_{gz}, b_{ax}, b_{ay}, b_{az}, g_{x}, g_{y}, g_{z}]$. We write these in short as:  $x = [p, v, \theta, b_{g}, b_{a}, g]$. Here,

- We are including gravity as a state varible because we want to estimate it what it is in our body frame real-time.
- We are including $b_g$ and $b_a$ because they we want to have a good, real-time estimate of that as well.

$$
\begin{gather*}
\begin{aligned}
& x = [p, v, R, b_a, b_g, g]
\end{aligned}
\end{gather*}
$$

As we have seen in the **motivating example,** there are our best estimates $x$, their "true values" $x_t$, and their differences (or errors) $\delta x$.  We also saw that **we use Kalman Filter to estimate the error $\delta x$**. So this relationship is:

$$
\begin{gather*}
&
p_t = p + \delta p  \tag{1}
\\ &
v_t = v + \delta v
\\ &
exp(\theta_t^{\land}) = exp(\theta^{\land})exp(\delta \theta^{\land})
\\ &
b_{gt} = b_g + \delta b_g
\\ &
b_{at} = b_a + \delta b_a
\\ &
g_t = g + \delta g
\\ &
\end{gather*}
$$

Also, from kinematics, we can find the motion model. Considering in real life, there are noises, the motion model for the **true values** must include them. $\eta_{ba}$ and $\eta_{bg}$ are noises to the biases `b_a` and `b_g`, which are different than the noises to `a` and `w`

$$
\begin{gather*}
\begin{aligned}
&
p' = v
\\ &
v' = R^T(\tilde{a} - b_a - \eta_a) + g
\\ &
R' = R[\tilde{w} - b_g - \eta_g]^{\land}
\\ &
b_a' = \eta_{ba}
\\ &
b_g' = \eta_{bg}
\\ &
g = 0
\end{aligned}
\end{gather*}
$$

Where the IMU measurements $\tilde{a}$, $\tilde{w}$ are control inputs $u_k$. Later, we can incorporate GPS readings as observations. Note that because this formulatin has $R^T(\tilde{a} - b_a - \eta_a)$ and $R[\tilde{w} - b_g - \eta_g]^{\land}$, **we can't write the above in the form of** `x' = Ax + Bu` yet. Currently it's still

$$
\begin{gather*}
\begin{aligned}
& x_{k+1} = f(x_k, u_k)
\end{aligned}
\end{gather*}
$$

### Why We Are Not Using The Conventional EKF

In **discrete time EKF**, we need to estimate the state covariance matrix so we can calculate the kalman gain, and our final updated state variables. The prediction is:

$$
\begin{gather*}
\begin{aligned}
& x_{k+1} =  f(x_k, u_k)
\\ &
 P_k^* = A_k^T P_{k-1} A_k + R \Rightarrow  P_k^* = \frac{\partial f}{\partial x} P_{k-1} \frac{\partial f^T}{\partial x} + R
\end{aligned}
\end{gather*}
$$

Here for the covariance matrix, we **must linearize the system to get the Jacobian $\frac{\partial f}{\partial x}$**. One way to do this is to use quaternion (or not-recommended, Euler-angles for gimbal lock).

However, **to use the SO(3) manifold perks, we stick to rotation matrices.** Now, we have a question: how do we get $\frac{\partial R}{\partial x}$? That will require derivative using perturbations. Without introducing tensors (multi-dimensional matrices), that'd be matrix-vector derivative, which is impossible. Another consideration for self driving cars is: if we are using global coordinate system, like UMT, or latitude-longitude, coordinates are relatively large numbers compare to the floating point number range.

### [Step 2] Continuous Time Error State-Space Model

The error $\delta x$ model is similar to the regular model

$$
\begin{gather*}
\begin{aligned}
& \delta p' = \delta v
\\ &
\delta b_g' = \eta_g
\\ &
\delta b_a' = \eta_a
\\ &
\delta g = 0
\end{aligned}
\end{gather*}
$$

The rotation and velocity errors are a bit involved because of the time derivative of `R`

$$
\begin{gather*}
\begin{aligned}
& R_t = R \delta R \rightarrow R_t' = R' \delta R + R \delta R' := R_t (\tilde{w} - b_{gt} - \eta_g)^{\land}
\\ &
\text{since:}
\\ &
R' = R (\tilde{w} - b_g)^{\land}
\\ &
\delta R = exp(\delta \theta^{\land})
\\ &
exp(\delta \theta^{\land})' = exp(\delta \theta^{\land})\theta^{\land}
\\ &
\Rightarrow
\\&
R (\tilde{w} - b_g)^{\land} exp(\delta \theta^{\land}) + R exp(\delta \theta^{\land}) (\delta \theta^{\land})' = R exp(\delta \theta)(\tilde{w} - b_{gt} - \eta_{g})^{\land}

\\ &
\text{using: } \phi^{\land}R = R(R^{T}\phi)^{\land}
\\ &
\Rightarrow [(\delta \theta)']^{\land} \approx (\tilde{w} - b_{gt} - \eta_g)^{\land} - [(I - \delta \theta)^{\land}(\tilde{w} - b_{gt})]^{\land}
\\&
\Rightarrow (\delta \theta)' \approx -(\tilde{w} - b_{g})^{\land} \delta \theta - \delta b_g - \eta_g
\end{aligned}
\end{gather*}
$$

During the above steps, we ignored secodnary infinitesmal values. Now we are ready for getting $\delta v'$, too:

$$
\begin{gather*}
\begin{aligned}
& v_t' = v' + \delta v' := R_t(\tilde{a} - b_{at} - \eta_a) + g_t

\\ &
\rightarrow v' + \delta v' := R(\tilde{a} - b_a) + g + \delta v'

\\ &
\rightarrow R_t(\tilde{a} - b_{at} - \eta_a) + g_t = R exp(\delta \theta)(\tilde{a} - b_{a} - \delta b_a - \eta_a) + g + \delta g
\\ & \approx R(I + \delta \theta^{\land} )(\tilde{a} - b_{a} - \delta b_a - \eta_a) + g + \delta g
\\ & \approx R \tilde{a} - Rb_{a} - R\delta b_a - R\eta_a + R\delta \theta^{\land} \tilde{a} - R\delta \theta^{\land}b_{a} + g + \delta g
\\ & = R \tilde{a} - Rb_{a} - R\delta b_a - R\eta_a - R\tilde{a}^{\land}\delta \theta +  Rb_{a}^{\land}\delta \theta + g + \delta g

\\ & \text{Using: }  R\eta_a = \eta_a
\\ &
\Rightarrow \delta v' = - R(\tilde{a} - b_a)^{\land}\delta \theta - R\delta b_a - \eta_a + \delta g
\end{aligned}
\end{gather*}
$$

Note that above, we ignored $ R\eta_a$ because $R^TR$ = I, $ R\eta_a$ is still zero-mean white Gaussian noise.

All together, in continuous time, the error $\delta x$ between our best estimate and the truth value has the motion model:

$$
\begin{gather*}
& \delta p' = \delta v
\\ &
\delta v' = - R(\tilde{a} - b_a)^{\land}\delta \theta - R\delta b_a - \eta_a + \delta g
\\ &
(\delta \theta)' \approx -(\tilde{w} - b_{g})^{\land} \delta \theta - \delta b_g - \eta_g
\\ &
\delta b_g' = \eta_{bg}
\\ &
\delta b_a' = \eta_{ba}
\\ &
\delta g = 0

\tag{2}
\end{gather*}
$$

### [Step 3] Discrete Time Error State Space Model

Using simple $\delta x_{k+1} = \delta x_k + \delta x_{k}' \Delta t$, we can write:

$$
\begin{gather*}
\begin{aligned}
& \delta p_{k+1} = \delta p_{k} + \delta v \Delta t

\\ &
\delta v_{k+1} = \delta v_{k} + (- R(\tilde{a} - b_a)^{\land}\delta \theta - R\delta b_a + \delta g) \Delta t - \eta_v
\\ &
(\delta \theta)_{k+1} \approx exp(-(\tilde{w} - b_{g}) \Delta t)\delta \theta - \delta b_g \Delta t - \eta_{\theta}
\\ &
\delta (b_g)_{k+1} = \delta (b_g)_{k} + \eta_{bg}
\\ &
\delta (b_a)_{k+1} = \delta (b_a)_{k} + \eta_{ba}
\\ &
\delta g_{k+1} = \delta g_{k}
\end{aligned}
\end{gather*}
$$

Why
$$
(\delta \theta)_{k+1} \approx exp(-(\tilde{w} - b_{g}) \Delta t)\delta \theta - \delta b_g \Delta t - \eta_{\theta}
$$?

- Because if in continuous time we have:

$$
\begin{gather*}
\begin{aligned}
& x' = Ax + Bu
\end{aligned}
\end{gather*}
$$

- Its discrete time counterpart is (**[PROOF SKIPPED]**):

$$
\begin{gather*}
\begin{aligned}
& x_{k+1} = exp^{A_k \Delta t}x_k + u_k \Delta t
\end{aligned}
\end{gather*}
$$

[From here,](./2024-03-22-robotics-imu-math.markdown) as a recap, the standard deviation of the noise terms are:

$$
\begin{gather*}
\begin{aligned}
& \sigma(\eta_v) = \Delta t \sigma(\eta_a)
\\ & \sigma(\eta_\theta) = \Delta t \sigma(\eta_w)
\\ & \sigma(\eta_{bg}) = \sqrt{\Delta t} \sigma_{bg}
\\ & \sigma(\eta_{ba}) = \sqrt{\Delta t} \sigma_{ba}
\end{aligned}
\end{gather*}
$$

### [Step 4] Discrete Time ESKF Motion Prediction

From equations `(2)`, the continuous system can be generically written with $f(\delta x)$, and Gaussian noise $n$

$$
\begin{gather*}
\begin{aligned}
& \delta x' = f(\delta x) + n
\end{aligned}
\end{gather*}
$$

- Noise is $n \sim \mathcal(0, Q)$

So it's easy to write out the motion prediction:

1. **Motion prediction** in ESKF is already linear, which is great!😊 **However, one thing to note is in ESKF, update $\delta x_{k+1}$ will be set to 0 after the update**, so this step is **optional:**

$$
\begin{gather*}

\delta x_{k+1}^* = F \delta x_{k}

\\
\Rightarrow
\\

\begin{bmatrix}
\delta p_{k+1}*\\
\delta v_{k+1}* \\
\delta \theta_{k+1}*\\
\delta  b_{g, k+1}* \\
\delta  b_{a, k+1}*\\
\delta g_{k+1}* \\
\end{bmatrix}

=

\begin{bmatrix}
I & I\Delta t & 0 & 0 & 0 &0 \\
0 & I\Delta t & -R(\tilde{a}-b_a)^{\land}\Delta t & 0 & -R \Delta t & I\Delta t \\
0 & 0 & exp(-(\tilde{w} - b_{g}) \Delta t) & -I \Delta t & 0 & 0 \\
0 & 0 & 0 & I & 0 & 0 \\
0 & 0 & 0 & 0 & I & 0 \\
0 & 0 & 0 & 0 & 0 & I \\
\end{bmatrix}

\begin{bmatrix}
\delta p_{k} \\
\delta v_{k} \\
\delta \theta_{k} \\
\delta  b_{g, k} \\
\delta  b_{a, k} \\
\delta g_{k} \\
\end{bmatrix}

\end{gather*}
$$

2. Covariance matrix is updated accordingly:

$$
\begin{gather*}
\begin{aligned}
& P_{k+1}* = F P_{k} F^T + Q
\end{aligned}
\end{gather*}
$$

### [Step 5] Discrete Time ESKF Observation Update

Observation `z` in continuous time is:

$$
\begin{gather*}
\begin{aligned}
& z(x) = h(x) + v
\end{aligned}
\end{gather*}
$$

Where the observation noise is: $v \sim \mathcal(0, V)$. We are using `v` because `R` is already in use :)

Then, we linearize `z` just like we do in regular EKF. Note that here our independent varables are $\delta x$:

$$
\begin{gather*}
\begin{aligned}
& z = H \cdot \delta x + v

\\ &
\Rightarrow
\\ &
H = \frac{\partial h}{\partial x} \frac{\partial x}{\partial \delta x}
\end{aligned}
\end{gather*}
$$

Because in continuous time, we have defined:

$$
\begin{gather*}
\begin{aligned}
& x_t = x \oplus \delta x
\\ &
\rightarrow
\\
&
p_t = p + \delta p
\\ &
v_t = v + \delta v
\\ &
exp(\theta_t^{\land}) = exp(\theta^{\land})exp(\delta \theta^{\land})
\\ &
b_{gt} = b_g + \delta b_g
\\ &
b_{at} = b_a + \delta b_a
\\ &
g_t = g + \delta g
\end{aligned}
\end{gather*}
$$

and we know **directly** from the observation model:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial h}{\partial x_t}
\end{aligned}
\end{gather*}
$$

We can know:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial h}{\partial x} |_{x=x_{k+1}} = \frac{\partial h}{\partial x_t} \frac{\partial x_t}{\partial \delta x}_{x=x_{k+1}}
\end{aligned}
\end{gather*}
$$

Where:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial x_t}{\partial \delta x} = [I_3, I_3, \frac{\partial Log(exp(\theta^{\land})exp(\delta \theta^{\land}))}{\partial \delta \theta}, I_3, I_3, I_3]
\end{aligned}
\end{gather*}
$$

- **One simplifying assumption here is: $\delta \theta$ is small enough to be a perturbation to $\theta$**. From [here](./2024-03-15-robotics-foundamentals-velocities.markdown), we can use the right perturbation and the BCH formula to get:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial Log(exp(\theta^{\land})exp(\delta \theta^{\land}))}{\partial \delta \theta} = J_r^{-1} \theta
\end{aligned}
\end{gather*}
$$

- If $\delta \theta$ is not small enough, we can write out the full form as well:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial Log(exp(\theta^{\land})exp(\delta \theta^{\land}))}{\partial \delta \theta} = J_r^{-1} Log(exp(\theta^{\land})exp(\delta \theta^{\land})) exp(\delta \theta^{\land})^T
\end{aligned}
\end{gather*}
$$

Kalman Gain and Covariance matrix updates stay the same:

$$
\begin{gather*}
\begin{aligned}
& K_{k+1} = P_{k+1}^{*} H_{k+1}^{T}(V^{-1} + H_{k+1} P_{k+1}^{*} H_{k+1}^T)

\\ &
P_{k+1} = P_{k+1}^{*} - K_{k+1} C_{k+1} P_{k+1}^{*} \Rightarrow P_{k+1} = P_{k+1}^{*} - K_{k+1} \frac{\partial h}{\partial x} P_{k+1}^{*}
\end{aligned}
\end{gather*}
$$

And
$$
\begin{gather*}
\begin{aligned}
& \delta x_{k+1} = K_{k+1} (z - h(x_{k+1}^{*}))
\end{aligned}
\end{gather*}
$$

### [Step 6] Discrete Time ESKF Final State Update and Error Reset

In discrete time, we approximate $p_{k+1}$ as the true value $p_t$ can define:

$$
\begin{gather*}
\begin{aligned}
& x_{k+1} = x_k \oplus \delta x_k
\\
\rightarrow
\\
& p_{k+1} = p_{k} + \delta p_{k}
\\ &
v_{k+1} = v_{k} + \delta v_{k}
\\ &
exp(\theta_{k+1}^{\land}) = exp(\theta_{k}^{\land})exp(\delta \theta_{k}^{\land})
\\ &
b_{g, k+1} = b_{g, k} + \delta b_{g, k}
\\ &
b_{a, k+1} = b_{a, k} + \delta b_{a, k}
\\ &
g_{k+1} = g_{k} + \delta g_{k}
\end{aligned}
\end{gather*}
$$

Since we have applied a correction, we can go ahead and reset $\delta x = 0$

**However, we recognize that this correction may not update with the best reset.** So, we need to adjust the error covariance before proceeding to the next step. We assume that after the reset $\delta x_{k}$, there's still an remeniscent error $\delta x^+$

The reset is to correct $x_{k+1} \sim \mathcal(\delta x, P_{k})$ to $x_{k+1} \sim \mathcal(0, P_{reset})$. **For vector space variables `p, v, b_a, b_g, g` this reset is a simple shift of distribution. The covariance matrices stay the same.** For rotation variables $\theta$ though, this shift of distribution is in the tanget space (which is a vector space). But projected on to the `SO(3)` manifold, the distribution is not only shifted, but also scaled. So, to find the new covariance matrix:

$$
\begin{gather*}
\begin{aligned}
& exp(\delta \theta) = exp(\delta \theta_k ) exp(\delta \theta^+ )

\\ &
\rightarrow exp(\delta \theta^+ ) = (-\delta \theta_k )exp(\delta \theta)
\\&
\text{Using BCH:}
\theta^+ \approx -\delta \theta_k + \delta \theta - \frac{1}{2} \delta \theta_k^{\land} \delta \theta + o((\delta \theta_k )^2)

\\ &
\rightarrow \frac{\partial \theta^+}{\partial \delta \theta} = I-\frac{1}{2} \delta \theta_k^{\land}
\end{aligned}
\end{gather*}
$$

Then, the overall "Jacobian" for the covariance is:

$$
\begin{gather*}
\begin{aligned}
& J_k = [I_3, I_3, I-\frac{1}{2} \delta \theta_k^{\land}, I_3, I_3, I_3]
\end{aligned}
\end{gather*}
$$

So the covariance reset is:

$$
\begin{gather*}
\begin{aligned}
& P_{reset} = J_k P_{k+1} J_k
\end{aligned}
\end{gather*}
$$

**Usually, this is close enough to identity because the $\theta$ covariance is small**

TODO: Is this linear BCH? why do we use jacobian here?

### [Step 7] GNSS Fusion

## A Quick Summary

The main differences between ESKF and EKF is:

- ESKF's motion update is already linearized during the velocity and angular value! No extra linearization is needed
- Kalman Filtering is applied on the error between the estimates and the true values, not on the estimates directly.
- The use of generic + ($\oplus$) for updating motion model and observation with SO(3) manifold
- In ESKF, we need to reset $\delta x = 0$ and $P_{k+1} = J_k P_{k+1} J_k$

$$
\begin{gather*}
\begin{aligned}
& \text{true value = state variable on a manifold + error in tanget space (zero-mean Gaussian distribution)}
\end{aligned}
\end{gather*}
$$

## Appendix - RTK GPS

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/0750cb08-5a84-4902-abb6-646b31d61110" height="300" alt=""/>
       </figure>
    </p>
</div>
