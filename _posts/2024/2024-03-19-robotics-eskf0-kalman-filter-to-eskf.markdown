---
layout: post
title: Robotics - An Overview Of Kalman Filter, Extended Kalman Filter, and Error State Kalman Filter
date: '2024-03-15 13:19'
subtitle: 
comments: true
tags:
    - Robotics
---

## [1] What Does A Kalman Filter Do?

If a system is a linear system, with Gaussian measurement and control noise, the Kalman Filter (KF) produces the minimum-covariance (i.e., the mean squared error) of the system states, given past control variables and sensor readings.

$$
\begin{gather*}
\begin{aligned}
\mathbf x_k &= \mathbf A\,\mathbf x_{k-1} + \mathbf B\,\mathbf u_k + \mathbf w_k, &\quad \mathbf w_k &\sim \mathcal N(\mathbf 0,\,\mathbf Q) \\
\mathbf z_k &= \mathbf C\,\mathbf x_k             +             \mathbf v_k, &\quad \mathbf v_k &\sim \mathcal N(\mathbf 0,\,\mathbf R)
\end{aligned}
\end{gather*}
$$

- where $\mathbf w_k$ is process noise and $\mathbf v_k$ is measurement noise; both are assumed zero‑mean, white and mutually independent.

**The Kalman Filter seeks to generate a sequence of estimates $\hat{x_k}$ such that its error covariance w.r.t the ground truth is minimized.**

$$
\begin{gather*}
\begin{aligned}
& \mathbf P_k \;=\; \operatorname{E}\big[(\hat{\mathbf x}_k-\mathbf x_k)(\hat{\mathbf x}_k-\mathbf x_k)^\top\big] \;\to\; \text{min.}
\end{aligned}
\end{gather*}
$$

1. With currently known control command $u_k$, we can make a prediction $\hat{\mathbf x}_k^-$.

$$
\begin{gather*}
\begin{aligned}
& \hat{\mathbf x}_k^- = \mathbf A \hat{\mathbf x}_{k-1}^- + \mathbf B\mathbf u_k
\end{aligned}
\end{gather*}
$$

- Of course, this prediction comes with some uncertainty. The error covariance is amplified to:

$$
\begin{gather*}
\begin{aligned}
& \mathbf P_k^- = \mathbf A\mathbf P_{k-1}\mathbf A^\top + \mathbf Q
\end{aligned}
\end{gather*}
$$

2. **By design, the Kalman Filter applies a correction to the prediction** (a.k.a prior) $\hat{\mathbf x}_k^-$, using the measurement $z_k$ with Kalman gain $K_k$

$$
\begin{gather*}
\begin{aligned}
& \hat{\mathbf x}_k = \hat{\mathbf x}_k^- + \mathbf K_k\,(\mathbf z_k - \mathbf C\hat{\mathbf x}_k^-),
\end{aligned}
\end{gather*}
$$

- So now, define the error term $e_k$, and transform it to a function of $K_k$, measurement noise $v_k$ and observation matrix $C$:

$$
\begin{gather*}
\begin{aligned}
& \begin{aligned}
\mathbf e_k &= \mathbf x_k - \hat{\mathbf x}_k \\
            &= (\mathbf I - \mathbf K_k\mathbf C)\,(\mathbf x_k - \hat{\mathbf x}_k^-) - \mathbf K_k\mathbf v_k.
\end{aligned}
\end{aligned}
\end{gather*}
$$

- Further, because the noise $v_k$ is independent from the state $x_k$ because we can rewrite the error covariance using $K_k$ as (skipping some steps here):

$$
\begin{gather*}
\begin{aligned}
& \mathbf P_k = (\mathbf I-\mathbf K_k\mathbf C)\,\mathbf P_k^-\,(\mathbf I-\mathbf K_k\mathbf C)^\top + \mathbf K_k\mathbf R\,\mathbf K_k^\top.  \tag{★}
\end{aligned}
\end{gather*}
$$

3. **Now, we are in a good place to solve for $K_k$, such that $P_k$ is at a local minima.** This requires the partial derivative of each covariance term w.r.t $K_k$ being zero:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial\,\operatorname{tr}(\mathbf P_k)}{\partial\,\mathbf K_k}=0 \;\Longrightarrow\;
\boxed{\;\mathbf K_k = \mathbf P_k^-\,\mathbf C^\top\,(\mathbf C\,\mathbf P_k^-\,\mathbf C^\top + \mathbf R)^{-1}\;}.  \tag{Kalman Gain}
\end{aligned}
\end{gather*}
$$

- This way, $K_k$ minimizes each covariance term, hence the trace. $K_k$ is now called the optimal gain.

- Substituting the optimal gain back into (★) yields the familiar closed‑form covariance update

$$
\begin{gather*}
\begin{aligned}
& \mathbf P_k = (\mathbf I-\mathbf K_k\mathbf C)\,\mathbf P_k^-.
\end{aligned}
\end{gather*}
$$
