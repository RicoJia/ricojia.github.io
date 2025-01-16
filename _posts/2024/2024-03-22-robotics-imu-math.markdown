---
layout: post
title: Robotics - [ESKF Series 1] IMU Model 
date: '2024-03-22 13:19'
subtitle: Specific Force, IMU model
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
  - So $R^T = R_{bw}$ is the car-to-world rotation


## The IMU Kinematics Model Set Up

Assumptions:

- The general frame of reference is: XYZ = forward, left, up.
- On earth, when free-falling, the IMU cannot detect gravity. When stationary, the IMU can measure the support `+g`. So the IMU usually measures the "specific force". Some IMU could omit `+g` though.
- In a car, we can measure the actual acceleration (in free space) $\tilde{a}$ and angular velocity $\tilde{w}$ w.r.t their world frame counterparts. **Here we assume the world is flat**
- Here, we **assume the IMU is mounted at the car's center of mass**. Otherwise, the imu could sense the Coriolis force, angular acceleration, and the centrifugal force.
- In reality, vibrations from the suspension of a car could be detected too.

The specific force kinematics model is:

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

## Continuous Time IMU Kinematics Model

IMU is susceptible to **noise** and **bias**.

- Even when stationary, the IMU does **NOT** have a zero-mean white noise on `a`, `w`. So we add mathematical bias terms $b_a$, $b_g$ to characterize it. **Note that this bias is affected by temperature, even. It's NOT a physical property, yet just a mathematical simplification.** The bias is a Wiener Process, whose time derivative is a Gaussian Process (This is also a Brownian motion, or random walk).
- The noise itself, $\eta_a$, $\eta_g$ is a zero-mean Gaussian random process

[For more about Gaussian Process and Power Spectral Density, please check here](../2017/2017-06-03-stats-basic-recap.markdown)

$$
\begin{gather*}
\begin{aligned}
& \tilde{a} = R^T(a - g) + b_a + \eta_a

\\
& \tilde{w} = w + b_g + \eta_g
\end{aligned}
\end{gather*}
$$

Where the biases' time derivatives are zero-mean Gaussian random processes, with **covariance functions** $\sigma_{ba}$ and $\sigma_{bg}^2$:

$$
\begin{gather*}
\begin{aligned}
& b_a'(t) \sim \mathcal{gp}(0, \sigma_{ba}^2 \delta(t - t') )
\\
& b_g'(t) \sim \mathcal{gp}(0, \sigma_{bg}^2 \delta(t - t') )

\\
& \eta_a \sim \mathcal{gp}(0, \sigma_{\eta a}^2 \delta(t - t') )
\\
& \eta_g \sim \mathcal{gp}(0, \sigma_{\eta g}^2 \delta(t - t') )
\end{aligned}
\end{gather*}
$$

- The covariance functions of $b_a(t)$ and $b_g(t)$ increases over time. So the IMU measurements $\tilde{a}$ and $\tilde{w}$ will become less accurate over time.
- The bias appears to be in Brownian motion (random walk). The higher range of the random walk, the "less stable" we call the bias.
- **A good IMU should have a bias relatively stable around 0**

## Discrete Time IMU Kinematics Model

The derivation of the Discrete time IMU kinematics model is quite lengthy. For those who are curious, please check out the appendix of [this paper](http://www.acsu.buffalo.edu/~johnc/gpsins_gnc05.pdf). The summary is:

- The discrete time value of $\tilde{w}(t + \Delta t)$ is the time average of the integral of $\tilde{w}$ over $\Delta t$. Over $\Delta t$, we assume that the IMU true value $w(t)$ is constant.

$$
\begin{gather*}
\begin{aligned}
& \tilde{w}(t + \Delta t) = \frac{1}{\Delta t}\int_{t_0}^{t_0 + \Delta t} [w(t) + b_g(t) + \eta_g(t)] dt
\end{aligned}
\end{gather*}
$$

The final covariances for the noises are:

$$
\begin{gather*}
\begin{aligned}
& \eta_g(k) \sim \mathcal{N(0, \frac{1}{\Delta t}Cov(\eta_g))}

\\ & \eta_a(k) \sim \mathcal{N(0, \frac{1}{\Delta t}Cov(\eta_a))}
\end{aligned}
\end{gather*}
$$

The final covariances for the biases are:

$$
\begin{gather*}
\begin{aligned}
& b_g(k) \sim \mathcal{N(0, \Delta t Cov(b_g))}

\\ & b_a(k) \sim \mathcal{N(0, \Delta t Cov(b_a))}
\end{aligned}
\end{gather*}
$$

### Units of IMU Covariances

The discrete time standard deviations can be derived from their continuous counterparts from the above:

$$
\begin{gather*}
\begin{aligned}
& \sigma_g(k) = \frac{1}{\sqrt{\Delta t}}\sigma_g
\\ & \sigma_a(k) = \frac{1}{\sqrt{\Delta t}}\sigma_a
\\ & \sigma_{bg}(k) = \sqrt{\Delta t} \sigma_{bg}
\\ & \sigma_{ba}(k) = \sqrt{\Delta t} \sigma_{ba}
\end{aligned}
\end{gather*}
$$

In discrete time, the standard deviation of the biases and the noises can be added right on to the state variables. So they share the same units??

$$
\begin{gather*}
\begin{aligned}
& \sigma_g(k) = \sigma_{bg}(k) = rad/s
\\ & \sigma_{ba}(k) = \sigma_a(k) = m/s^2
\end{aligned}
\end{gather*}
$$

Accordingly, in continuous time, one can derive:

$$
\begin{gather*}
\begin{aligned}
& \sigma_g = \frac{rad}{\sqrt{s}}
\\ & \sigma_{bg} = \frac{rad}{s\sqrt{s}}
\\ & \sigma_a = \frac{m}{s\sqrt{s}}
\\ & \sigma_{ba} = \frac{m}{s^2\sqrt{s}}
\end{aligned}
\end{gather*}
$$

In some IMU documentations, people use $\frac{1}{\Delta t} = Hz$, or using $1/\sqrt{hour}$. 

- E.g., $\sigma_g = 0.66^\circ/\sqrt{hour}$ and $\sigma_a = 0.11 m/s/\sqrt{hour}$ means with the correct biases, within an hour, the IMU will have an integration error of $0.66 ^\circ$ and $0.11m/s$. (Note they are angle and velocity, first degree integration)
- IMU documentations typically do NOT have bias covariances $\sigma_{bg}$, $\sigma_{ba}$ because in real life they are hard to measure? Also, IMU bias covariances are usually **affected by temperature** as well. So, we usually need to estimate them real-time.

### Simple IMU Integration Using Recursion

At a single timestep, we take into account the accleration in a reference frame

$$
\begin{gather*}
\begin{aligned}
& p(t + \Delta t) = p(t) + v(t) \Delta t + \frac{1}{2}[R(t) (\tilde{a} - b_a) + g]^2 \Delta t^2

\\ & 
R(t + \Delta t) = R(t) Exp((\tilde{w} - b_g) \Delta t)

\\ &
v(t + \Delta t) = v(t) + R[\tilde{a} - b_a + g] \Delta t
\end{aligned}
\end{gather*}
$$

If we want to integrate the IMU readings, we can the Euler Method Integration recursively (instead of higher order estimations like Runge-Kutta methods):

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/15286927-4286-47e6-a711-7539a9366f2e" height="200" alt=""/>
    </figure>
</p>
</div>

$$
\begin{gather*}
\begin{aligned}
& p(t) = p_0 + \sum_{n=0}^{k-1} v(t)\Delta t + \sum_{n=0}^{k-1} \frac{1}{2} R(n)[\tilde{a}_n - b_{a, n} + g_{n}]^2 \Delta t^2
\\ & 
R(t) = R(0) \prod_{n=0}^{k-1} Exp((\tilde{w_n} - b_{g,n}) \Delta t)
\\ &
v(t) = v(0) + \sum_{n=0}^{k-1} R(n)[\tilde{a}_n - b_{a, n} + g_{n}] \Delta t
\end{aligned}
\end{gather*}
$$

The location integration will quickly diverge from the ground truth because it is second order integration.

## IMU Initialization

The IMU Model of accleration and angular velocity are: 

$$
\begin{gather*}
\begin{aligned}
& \tilde{a} = R^T(a - g) + b_a + \eta_a

\\
& \tilde{w} = w + b_g + \eta_g
\end{aligned}
\end{gather*}
$$

So if we set the IMU idle for 10s (so `w=0`, `a`=0), we will collect multiple $\tilde{w}$ and $\tilde{a}$. So, since we assume Gaussian noises, and `R=I`,

$$
\begin{gather*}
\begin{aligned}
& mean(\tilde{w}) = b_g
\\ &
\tilde{a} = b_a - g

\end{aligned}
\end{gather*}
$$

For $a$, we do:

$$
\begin{gather*}
\begin{aligned}
& g = mean(\tilde{a}) / |mean(\tilde{a})| * 9.8
\\ & 
b_a = mean(\tilde{a}) - g
\end{aligned}
\end{gather*}
$$