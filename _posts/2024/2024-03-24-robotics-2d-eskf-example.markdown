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

## Motivating Example.

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
= Exp(\delta \theta)(\tilde{w} - \eta_g)^{\land} - Exp(\delta \theta)[\tilde{w}]^{\land} 

\\&
= -Exp(\delta \theta)[\eta_g]^{\land}

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

### Motion Model All Together

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

## Observation Model