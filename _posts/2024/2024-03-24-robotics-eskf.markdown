---
layout: post
title: Robotics - Error-State Kalman Filter (ESKF ) in GNSS-Inertial Navigation System (GINS)
date: '2024-03-24 13:19'
subtitle: ESKF, GINS
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Motivating Example

Consider a 2D point travelling with constant velocity. $x = [p_x, p_y, v_x, v_y]$. Within ESKF, we think that compared to the true state values $x_t = [p_{xt}, p_{yt}, v_{xt}, v_{yt}]$, our state variable estimates $x = [p_x, p_y, v_x, v_y]$ will be subject to Gaussian errors.: $\delta x = [\delta p_x, \delta p_y, \delta v_x, \delta v_y]$

The motion update is:

$$
\begin{gather*}
\begin{aligned}
& x_{k+1}^* = x_{k} +
\begin{bmatrix}
p_x \\ p_y \\ v_x \\ v_y
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & \Delta t & 0  \\
0 & 1 & & 0 & \Delta t  \\
0 & 0 & & 1 & 0  \\
0 & 0 & & 0 & 1  \\
\end{bmatrix}

\begin{bmatrix}
p_x' \\ p_y' \\ v_x' \\ v_y'
\end{bmatrix}

= x_{k} +  Fx'_{k}

\end{aligned}
\end{gather*}
$$

In the meantime, we can update the covariance matrix as usual: $P_{k+1}^* = A_{k+1}^T P_{k} A_{k+1} + Q$

**The update is:**

$$
\begin{gather*}
\begin{aligned}
& \delta x_{k+1} = K_{k+1} (z_{k+1} - Hx^*_{k+1})

x_{k+1} = x_{k+1}^* \oplus \delta x_{k+1}
\end{aligned}
\end{gather*}
$$

Here, $\oplus$ is the "generic add", which "adds" on a manifold.

## ESKF in GINS (GPS-Intertial Navigation System)

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

### Why We Are Not Using The Conventional EKF**

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
\text{using: } \phi^{\land} = R(R^{T}\phi)^{\land}
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

\delta x_{k+1}* = F \delta x_{k}

\\ 
\Rightarrow
\\

\begin{bmatrix}
\delta p_{k+1}* \\
\delta v_{k+1}* \\
\delta \theta_{k+1}* \\
\delta  b_{g, k+1}* \\
\delta  b_{a, k+1}* \\
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



## A Quick Summary

The main differences between ESKF and EKF is:

- ESKF's motion update is already linearized during the velocity and angular value! No extra linearization is needed
- Kalman Filtering is applied on the error between the estimates and the true values, not on the estimates directly.
- The use of generic + ($\oplus$) for updating motion model and observation with SO(3) manifold

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
