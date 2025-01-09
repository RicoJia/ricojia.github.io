---
layout: post
title: Robotics - [ESKF Series 3] Motivational Example Of Error-State Kalman Filter (ESKF) 
date: '2024-03-24 13:19'
subtitle: ESKF
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Motivating Example

We use the same example as the one in [the EKF post](./2024-03-21-robotics-foundamentals-extended-kalman-filter.markdown): imagine our world has two cones with known locations: `c1`, `c2`, and a diff drive with wheel encoders and a cone detector.

- The encoder can give us body-frame linear and angular velocities $\tilde{v}$, $\tilde{w}$. They can be modelled as 'true value + noise':

$$
\begin{gather*}
\begin{aligned}
& \tilde{v} = v + \eta_v
\\ & \tilde{w} = w + \eta_g
\end{aligned}
\end{gather*}
$$
    - Note, we use $\eta_g$ to represent angular velocity noise

- The cone detector can give us the range and bearing of each cone from the robot: `d`, $\beta$. **The goal is to estimate the $[x, y, \theta]$ of the vehicle.**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/54cc023f-4c50-46ac-9161-6de47997d0b9" height="300" alt=""/>
        <figcaption><a href="https://yonan.org/ukal/examples/ekf_example.html">Source</a></figcaption>
    </figure>
</p>
</div>

### Motion Model

The motion model in EKF is:

$$
\begin{gather*}
\begin{aligned}
& x_{t+1} = x_t - \frac{\omega}{v} \sin(\theta_t) + \frac{\omega}{v} \sin(\theta_t + \omega \Delta t) + \eta_x, \\
& y_{t+1} = y_t + \frac{\omega}{v} \cos(\theta_t) - \frac{\omega}{v} \cos(\theta_t + \omega \Delta t) + \eta_y, \\
& \theta_{t+1} = \theta_t + \omega \Delta t + \eta_\theta.

\\ & \Rightarrow
\\ & x_{t+1} = f(x_t, u, \eta)
\end{aligned}
\end{gather*}
$$

In ESKF, we define an error $\delta x$ to be the difference between last state estiate $x_t$ and the true value of the current state $x_{t+1, true}$:

$$
\begin{gather*}
\begin{aligned}
& x_{t+1, true} = x_{t} + \delta x
\end{aligned}
\end{gather*}
$$

For the sake of convenience, we represent position `[x, y]` as `p`. So our state vector is $x = [p, \theta]$. Therefore, we have the error defined as:

$$
\begin{gather*}
\begin{aligned}
& p_{t+1, true} = p_t + \delta p
\\ & \theta_{t+1, true} = \theta_t + \delta \theta
\end{aligned}
\end{gather*}
$$

Also, since we have heading $\theta$, it's easy to have:

$$
\begin{gather*}
\begin{aligned}
& R_{world to car} = R = Exp(\theta)
\end{aligned}
\end{gather*}
$$

### Linearize $\delta \theta'$

In ESKF, we apply Kalman Filtering on Linearized errors. To linearize $\delta \theta$ is, we first look at the true [2D rotation matrix](./2024-03-10-robotics-foundamentals-rotations.markdown):

$$
\begin{gather*}
\begin{aligned}
& R_{true} = R Exp(\theta)
\\ \Rightarrow
\\ & R_{true}' = R'Exp(\delta \theta) + RExp(\delta \theta)' := R_t(\tilde{w} - \eta_g)^{\land}
\end{aligned}
\end{gather*}
$$

Using:

$$
\begin{gather*}
\begin{aligned}
& Exp(\delta \theta)' = Exp(\delta \theta) [\delta \theta']^{\land}
\\ &
R' = R[\tilde{w}]^{\land}
\end{aligned}
\end{gather*}
$$

We have:

$$
\begin{gather*}
\begin{aligned}
& Exp(\delta \theta) [\delta \theta']^{\land} = Exp(\delta \theta)(\tilde{w} - \eta_g)^{\land} - [\tilde{w}]^{\land}Exp(\delta \theta)
\end{aligned}
\end{gather*}
$$

[In this post](./2024-03-10-robotics-foundamentals-rotations.markdown), we see that

$$
\begin{gather*}
\begin{aligned}
& \phi^{\land} R = R \phi^{\land}
\end{aligned}
\end{gather*}
$$

So

$$
\begin{gather*}
\begin{aligned}
& Exp(\delta \theta) [\delta \theta']^{\land} = Exp(\delta \theta)(\tilde{w} - \eta_g)^{\land} - [\tilde{w}]^{\land}Exp(\delta \theta)

\\&
= Exp(\delta \theta)(\tilde{w} - \eta_g)^{\land} - Exp[\delta \theta](\tilde{w})^{\land}

\\&
= -Exp[\delta \theta](\eta_g)^{\land}

\\
\Rightarrow
\\ &
\delta \theta' = -\eta_g
\end{aligned}
\end{gather*}
$$

### Linearize $\delta p'$

Using first order Taylor expansion $ Exp(\delta \theta) \approx I + \delta \theta$
$$
\begin{gather*}
\begin{aligned}
& p_{true}' = p' + \delta p' = R_{true} (\tilde{v} - \eta_v)

\\ &
p' = R \tilde{v}

\\ \Rightarrow
\\ &
p' + \delta p' \approx R(I + \delta \theta)(\tilde{v} - \eta_v)

\\ \Rightarrow
\\ &
\delta p' \approx R(\tilde{v} - \eta_v) \delta \theta - R \eta_v
\end{aligned}
\end{gather*}
$$

### Motion Model All Together & Motion Update

$$
\begin{gather*}
\begin{aligned}
& \delta p_{t+1} = \delta p_{t} + (R(\tilde{v} - \eta_v) \delta \theta_t) \Delta t- \eta_v
\\ &
\delta \theta_{t+1} = \delta \theta_{t}  - \eta_{\theta}
\end{aligned}
\end{gather*}
$$

- $R \eta_v $ has the same covariance as $\eta_v $
- $\sigma(\eta_{\theta}) = \sigma(\eta_{g}) \Delta t$
- $\delta \theta_{t+1}$ is derived by integrating $\delta \theta$

The above already is linear!! So putting it in matrix form:

$$
\begin{gather*}
\begin{aligned}
& F_t =
\begin{bmatrix}
I_2 & R(\tilde{v} - \eta_v) \Delta t    \\
0_1 & I_1
\end{bmatrix}

\end{aligned}
\end{gather*}
$$

And this ESKF motion model really is the linearized approximation of error:

$$
\begin{gather*}
\begin{aligned}
& \delta x_{t+1} = F \delta x_{t} + w, w \sim \mathcal(0, Q)
\end{aligned}
\end{gather*}
$$

This step is optional in practice because delta x gets reset after correction. But that's what ESKF really tries to do

Accordingly, the covariance matrix of the error becomes:

$$
\begin{gather*}
\begin{aligned}
& P_{t+1}^* = F_tPF_t^T + Q
\end{aligned}
\end{gather*}
$$

## Observation Model

In this example, we assume that the landmark positions are known. In general EKF and ESKF, the observations $[d, \beta]$ we get for each cone is the result of the non-linear model of the **true value** plus the noise.

$$
\begin{gather*}
\begin{aligned}
& z_{true} = h(x_{true}) = h(x^* + \delta x) + V, V \sim \mathcal(0, R)
\end{aligned}
\end{gather*}
$$

So to linearize it, since we know the predicted $x_{t+1}^*$ we do:

$$
\begin{gather*}
\begin{aligned}
& z_{t+1, true} = h(x^*) + H \delta x
\end{aligned}
\end{gather*}
$$

But H will be the Jacobian w.r.t $\delta x$

$$
\begin{gather*}
\begin{aligned}
& H = \frac{\partial h}{\partial x}\frac{\partial x}{\partial \delta x}
\end{aligned}
\end{gather*}
$$

In this 2D diff drive example, we know from [the EKF example](./2024-03-21-robotics-foundamentals-extended-kalman-filter.markdown) that

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial h}{\partial x} =

\begin{bmatrix}
\frac{x - c_x}{d} & \frac{y - c_y}{d} & 0 \\
\frac{c_y - y}{d^2} & \frac{x - c_x}{d^2} & -1
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

The tricky part is finding $\frac{\partial x_{true}}{\partial \delta x}$. Finding $\frac{\partial p_{true}}{\partial \delta p}$ is easy because $p_{true} = p + \delta p$:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial p_{true}}{\partial \delta p} = I_2
\end{aligned}
\end{gather*}
$$

Special part about ESKF is $\frac{\partial \theta_{true}}{\partial \delta \theta}$

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \theta}{\partial \delta \theta} = \frac{\partial Log(R Exp(\delta \theta))}{\partial \delta \theta}
\end{aligned}
\end{gather*}
$$

Since in `SO(2)` we have the convenient property of:

$$
\begin{gather*}
\begin{aligned}
& Log(R Exp(\delta \theta)) = [\theta + \delta \theta]
\end{aligned}
\end{gather*}
$$

Then it's quite easy to get:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial x}{\partial \delta x} = 1
\end{aligned}
\end{gather*}
$$

So all together,

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial h}{\partial \delta x} =

\begin{bmatrix}
\frac{x - c_x}{d} & \frac{y - c_y}{d} & 0 \\
\frac{c_y - y}{d^2} & \frac{x - c_x}{d^2} & -1
\end{bmatrix}

I_3

\end{aligned}
\end{gather*}
$$

## Correction

So, we can calculate the Innovation, Kalman Gain quite easily:

$$
\begin{gather*}
\begin{aligned}
& S = H P_{t+1}^{*} H^\top + R
\\ &
K = P_{t+1}^{*} H^\top S^{-1}
\\ &
\delta x_{t+1} = \delta x + K (z_t -  h(x_{t+1}^*))
\\ &
P_{t+1}=(I−KH)P_{t+1}^{*}
\end{aligned}
\end{gather*}
$$

However, we need to reset $\delta x$ every step:

$$
\begin{gather*}
\begin{aligned}
& \delta x_{t+1} = 0
\end{aligned}
\end{gather*}
$$

And that's why the above update step is equivalent to:

$$
\begin{gather*}
\begin{aligned}
& \delta x_{t+1} = \delta x + K (z_t -  h(x_{t+1}^*)) = K (z_t -  h(x_{t+1}^*))
\end{aligned}
\end{gather*}
$$

Then finally, we need to apply $\delta x$ back onto $x$. For general rotation, we need to apply $\oplus$.

$$
\begin{gather*}
\begin{aligned}
& x_{t+1} = x_{t+1}^* \oplus \delta x
\end{aligned}
\end{gather*}
$$

However, since in 2D, rotation manifold happens to be the same as its tangent space, we can safely do regular addition without worrying about rotation matrix / quaternion multiplication

$$
\begin{gather*}
\begin{aligned}
& & x_{t+1} = x_{t+1}^* + \delta x
\end{aligned}
\end{gather*}
$$
