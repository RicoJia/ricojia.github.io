---
layout: post
title: Robotics - IMU Math
date: '2024-07-11 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Introduction

The IMU is a very common localization device. One can find it in most electronics nowadays: cars, smart watches, phones, even soccer balls. Knowing how IMU works could be a long process. In this article, we just focus on the "simple math API" of the IMU - the IMU model, and how to build a simple ESKF based localization system with it

In this article and beyond,

- $\tilde{w}$, $\tilde{a}$ are measured values of angular velocity, and linear acceleration
- $R = R_{wb}$ We represent the world rotation matrix $R_{wb}$ as $R$.
    - So $R_{bw}$ is the car-to-world rotation

## IMU Model

### The Kinematics Model

Assumptions:

- The general frame of reference is: XYZ = forward, left, up. 
- On earth, when free-falling, the IMU cannot detect gravity. When stationary, the IMU can measure the support `+g`. Some IMU could omit `+g` though.
- In a car, we can measure the actual acceleration (in free space) $\tilde{a}$ and angular velocity $\tilde{w}$ w.r.t their world frame counterparts. **Here we assume the world is flat**
- Here, we **assume the IMU is mounted at the car's center of mass**. Otherwise, the imu could sense the Coriolis force, angular acceleration, and the centrifugal force. 
- In reality, vibrations from the suspension of a car could be detected too.

$$
\begin{gather*}
\begin{aligned}
& \tilde{a} = R^T (a - g)

\\
& \tilde{w} = w
\end{aligned}
\end{gather*}
$$

And the general kinematics **without considering gravity**:

$$
\begin{gather*}
\begin{aligned}
& R' = Rw^{\land}

\\
& p' = v

\\
& v' = a
\end{aligned}
\end{gather*}
$$


### The IMU Noise Model

IMU is susceptible to **noise** and **bias**. Even when stationary, the IMU does **NOT** have a zero-mean white noise on `a`, `w`. So we add mathematical bias terms $b_a$, $b_g$ to characterize it. **Note that this bias is affected by temperature, even. It's NOT a physical property, yet just a mathematical term.** The bias is a Wiener Process, whose time derivative is a Gaussian Process (This is also a Brownian motion, or random walk). The noise itself, $\eta_a$, $\eta_g$ is a zero-mean Gaussian random process

$$
\begin{gather*}
\begin{aligned}
& \tilde{a} = R^T(a - g) + b_a + \eta_a

\\
& \tilde{w} = w + b_g + \eta_g
\end{aligned}
\end{gather*}
$$

Where the biases' time derivatives are zero-mean Gaussian random processes, with **covariance functions** $\Sigma_a$ and $\Sigma_g$:

$$
\begin{gather*}
\begin{aligned}
& b_a'(t) \sim \mathcal{gp}(0, \Sigma_a \delta(t - t') )
\\
& b_g'(t) \sim \mathcal{gp}(0, \Sigma_g \delta(t - t') )
\end{aligned}
\end{gather*}
$$

