---
layout: post
title: Robotics Fundamentals - Extended Kalman Filter (EKF)
date: '2024-03-15 13:19'
subtitle: Kalman Filter Framework
comments: true
tags:
    - Robotics
---

## Welcome To EKF - A Diff Drive Example

EKF (Extended Kalman Filter) is widely applied in robotics to smooth sensor noises to get a better estimate of a car's position.

Now let's take a look at a diff (differential) drive example. 

Imaging our world has two cones: `c1`, `c2`, and a diff drive with wheel encoders and a cone detector.

- A wheel encoder tells us how far (in m/s) a wheel has travelled within a known time window.
- The cone detector can detect the range `d` and bearing $\beta$ of each cone at any given time.

**The goal is to estimate the $[x, y, \theta]$ of the vehicle.**

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/54cc023f-4c50-46ac-9161-6de47997d0b9" height="300" alt=""/>
        <figcaption><a href="https://yonan.org/ukal/examples/ekf_example.html">Source</a></figcaption>
    </figure>
</p>
</div>

## Mathematical Formulation

### Wheel Encoder

The wheel encoder is able to give us a noisy estimate of the linear $v$ and angular velocity $w$ of the robot:

$$
\begin{gather*}
\begin{aligned}
& v = \frac{l + r}{c}
\\ & w = \frac{l - r}{2}
\\ & R = \frac{2(l+r)}{c(r-l)}
\end{aligned}
\end{gather*}
$$

Where:

- $v$ is the linear velocity of the robot's center in `m/s`
- $w$ is the angular velocity of the robot's rotation in `rad/s`
- $l, r$ are the wheel encoder increments within time window $\Delta t$ in `m/s`.
- $c$ is the distance between the two wheels (wheel base).

One can also notice that $v = wR$ in this case!


Conventionally, we use $x$ to represent our state vector as:

$$
\begin{gather*}
\begin{aligned}
& x = 
\begin{bmatrix}
x \\ y \\ \theta
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

And we deem the linear and angular velocities as control input ,$u$:
$$
\begin{gather*}
\begin{aligned}
u = \begin{bmatrix}
v \\ w
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

### Motion Model

In **discrete time** $t+1 = t + \Delta t$, we assume the robot **follows a circular motion**. That is, it has a constant linear and angular velocity.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/5b7c08d2-2c1a-493c-91c1-10e17449957d" height="300" alt=""/>
        <figcaption><a href="https://study.com/academy/lesson/instantaneous-uniform-angular-velocity-of-circular-motion.html">Source</a></figcaption>
    </figure>
</p>
</div>

Accordingly,

$$
\begin{gather*}
\begin{aligned}
& x' = v cos(\theta)
\\ & y' = v sin(\theta)
\\ & \theta' = w
\end{aligned}
\end{gather*}
$$

If we integrate the above derivatives, we can get the motion update. This motion update is equivalent to **gettting the new coordinates on a circle**. In the real world, we have a gaussian noise $\eta$ from the control update. 

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

- The process noise has a covariance matrix $Q$:

$$
\begin{gather*}
\begin{aligned}
& Q = 
\begin{bmatrix}
var(v) & 0  \\
0 & var(\theta)
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

However, **to do [Kalman Filtering](./2024-03-20-robotics-foundamentals-kalman-filter.markdown)**, we must be able to compute kalman gain and covariance matrix using a linear form:

$$
\begin{gather*}
\begin{aligned}
& x_{t + 1} = F_t x_t + B_t u_t + \eta
\end{aligned}
\end{gather*}
$$

However, because of the non linear terms like $\frac{w}{v}sin(\theta)$, we can't write the above into a linear form directly. However, we can use the taylor expansion to linearize it. One tricky thing is, the linearization is around an estimate. So the EKF framework is formulated **about the deviation from the true value.**

1. We first get a prediction $x_{t+1}^*$ using the full non-linear update model above, 
2. Define Deviation between the last state estimate and the next state estimate:

$$
\begin{gather*}
\begin{aligned}
&\delta x := x_{t+1, true} - \bar{x_{t}}
\end{aligned}
\end{gather*}
$$
2. Linearize the deviation $\delta x$

$$
\begin{gather*}
\begin{aligned}
& x_{t+1}^* = f(\bar{x_t}, u_t)

\\ \Rightarrow

\\ &
x_{t+1, true} = f(\bar{x_t} + \delta x) \approx x_{t+1}^* + J_x (x_{t + 1} - \bar{x_t}) = J_x x_{t + 1} + b
\end{aligned}
\end{gather*}
$$

So $x_{t+1, true} = J_x x_{t + 1} + b$ is linear, **w.r.t**  $x_{t+1}$! This is the foundamental difference from ESKF, where a linear system is built on error $\delta x = x_{t + 1, true} - x_{t+1}^*$

- This is because we our motion model estimates a single state estimate $x_{t+1}$, and still linearizes around $x_{t}$. However ESKF estimates the error $\delta x = x_{t + 1, true} - x_{t+1}^*$

$J_x$ is the Jacobian of $\frac{\partial f}{\partial x}$ 

$$
\begin{gather*}
\begin{aligned}
x_{t+1} \approx 

\begin{bmatrix}
1 & 0 & -\frac{w}{v} \cos(\theta_t) + \frac{w}{v}\cos(\theta_t + \Delta t) \\
0 & 1 & -\frac{w}{v} \sin(\theta_t) + \frac{w}{v}\sin(\theta_t + \Delta t) \Delta t \\
0 & 0 & 1
\end{bmatrix}
x_{t + 1} + b

\end{aligned}
\end{gather*}
$$


To be consistent with most other literature, we use $F_t$ to represent this Jacobian (a.k.a Prediction Jacobian)

$$
\begin{gather*}
\begin{aligned}
& F_t = J_x = \begin{bmatrix}
1 & 0 & -\frac{w}{v} \cos(\theta_t) + \frac{w}{v}\cos(\theta_t + \Delta t) \\
0 & 1 & -\frac{w}{v} \sin(\theta_t) + \frac{w}{v}\sin(\theta_t + \Delta t) \Delta t \\
0 & 0 & 1
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

### Observation Model

We assume that we know the positions of each landmark $[c_x, c_y]$
A cone appears to be `\beta, d` in our observation. Note that in real life, this observation is subject to observation noise:

$$
\begin{gather*}
\begin{aligned}
& d = \sqrt{(c_x - x)^2 + (c_y - y)^2} + \eta_{od}
\\ & \beta = atan2(\frac{y_c - y_1}{x_c - x_1}) - \theta + \eta_{ob}
\end{aligned}
\end{gather*}
$$

Similarly, we think that deviation of the predicted observation from the real observation comes from the deviation of state estimate $x_{t+1}^*$ and the true state: $x_{t+1, true}$. Using Taylor Expansion to linearize the observation model:

$$
\begin{gather*}
\begin{aligned}
& h(x_{t+1, true}) \approx h(x_{t+1}^*) + h(x)'|_{x=x_{t+1}^*}(x_{t+1} - x_{t+1}^*) 

\\ & =  h(x_{t+1}^*) + H(x) (x_{t+1} - x_{t+1}^*) 

\\ & 
\Rightarrow
\\ & 
h(x_{t+1, true}) - h(x_{t+1}^*) = z_t - h(x_{t+1}^*) = H(x) (x_{t+1} - x_{t+1}^*)
\end{aligned}
\end{gather*}
$$

- $h(x_{t+1, true}) := z_t$ is the real observation

Linearizing the above gives us the observation Jacobian $H$:

$$
\begin{gather*}
\begin{aligned}
& H = \frac{\partial h(x)}{\partial x}|_{x=x_{t+1}^*} = \begin{bmatrix}
\frac{x - c_x}{d} & \frac{y - c_y}{d} & 0 \\
\frac{c_y - y}{d^2} & \frac{x - c_x}{d^2} & -1
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

This is another difference from ESKF, where 

$$
\begin{gather*}
\begin{aligned}
& H = \frac{\partial h(x)}{\partial \delta x} =  \frac{\partial h(x)}{\partial x} \frac{\partial x}{\partial \delta x}
\end{aligned}
\end{gather*}
$$

- This is because we our observation model is still linearized on $x$ (at $x=x_{t+1}^*$), but ESKF's observation model views the entire $h(x_{t+1, true}) - h(x_{t+1}^*)$ caused by the error $\delta x$

## Filtering Process

In EKF, we are implicitly applying the Kalman Filtering on the devivation $\delta x$ from true state values. By applying kalman filtering, we can minimize the error covariance on the deviation given observation.

- Prediction is 

$$
\begin{gather*}
\begin{aligned}
& x_{t+1}^* = f(\bar{x_{t}}, u_t) = 

\begin{bmatrix}
x_t - \frac{\omega}{v} \sin(\theta_t) + \frac{\omega}{v} \sin(\theta_t + \omega \Delta t) \\
y_t + \frac{\omega}{v} \cos(\theta_t) - \frac{\omega}{v} \cos(\theta_t + \omega \Delta t) \\
\theta_t + \omega \Delta t

\\
\end{bmatrix}

\\ &
P_{t+1}^{*} = F_t P_t F_t^\top + Q,
\end{aligned}
\end{gather*}
$$

- Update is: 

$$
\begin{gather*}
\begin{aligned}
& S = H P_{t+1}^{*} H^\top + R
\\ & 
K = P_{t+1}^{*} H^\top S^{-1}
\\ & 
\bar{x_{t+1}} = x_{t+1}^* + K (z_t -  h(x_{t+1}^*))
\\ &
P_{t+1}=(I−KH)P_{t+1}^{*}
\end{aligned}
\end{gather*}
$$