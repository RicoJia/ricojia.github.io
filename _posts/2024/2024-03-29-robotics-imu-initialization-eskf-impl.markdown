---
layout: post
title: Robotics - [ESKF Series 6] IMU Initialization ESKF Implementation
date: '2024-03-29 13:19'
subtitle: ESKF
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

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