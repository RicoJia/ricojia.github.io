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

## [2] What Does An Extended Kalman Filter Do?

A system is non-linear when its state is a non-linear result of control signals , or when observations is a non-linear transform of the current state. (Its noise can still be Gaussian, mutually independent)

$$
\begin{gather*}
\begin{aligned}
\mathbf x_k &= f\bigl(\mathbf x_{k-1},\,\mathbf u_k\bigr) \, + \, \mathbf w_k, &\quad \mathbf w_k &\sim \mathcal N(\mathbf 0,\,\mathbf Q_k) \\
\mathbf z_k &= h\bigl(\mathbf x_k\bigr) \, + \, \mathbf v_k, &\quad \mathbf v_k &\sim \mathcal N(\mathbf 0,\,\mathbf R_k)
\end{aligned}
\end{gather*}
$$

However, if the non‑linearity is mild we can linearise the models around the current mean estimate. Then we are able to calculate the Kalman gain, and use that to apply a correction with an observation onto our prediction. Here is how:

1. Make prediction using the non-linear model:

$$
\begin{gather*}
\begin{aligned}
\hat{\mathbf x}k^- = f\bigl(\hat{\mathbf x}{k-1},,\mathbf u_k\bigr)
\end{aligned}
\end{gather*}
$$

2. The equivalent linearized non-linear model is:

$$
\begin{gather*}
\begin{aligned}
& \begin{aligned}
f(\mathbf x_{k-1},\mathbf u_k) &\approx f(\hat{\mathbf x}_{k-1},\mathbf u_k)
                  + \mathbf F_k\,\bigl(\mathbf x_{k-1} - \hat{\mathbf x}_{k-1}\bigr), \\
h(\mathbf x_k) &\approx h(\hat{\mathbf x}_k^-)
                  + \mathbf H_k\,\bigl(\mathbf x_k - \hat{\mathbf x}_k^-\bigr),
\end{aligned}
\end{aligned}
\end{gather*}
$$

- where $F_k$ and $H_k$ are Jacobian matrices:

$$
\begin{gather*}
\begin{aligned}
\mathbf F_k \triangleq \left.\frac{\partial f}{\partial \mathbf x}\right|_{\hat{\mathbf x}_{k-1},\,\mathbf u_k},
\qquad
\mathbf H_k \triangleq \left.\frac{\partial h}{\partial \mathbf x}\right|_{\hat{\mathbf x}_k^-}.
\end{aligned}
\end{gather*}
$$

3. After this linearisation step the model looks linear–Gaussian, so we can reuse the Kalman machinery with $\mathbf F_k$ and $\mathbf H_k$ playing the roles of $\mathbf A$ and $\mathbf C$:

$$
\begin{gather*}
\begin{aligned}
\text{Prior covariance update: } \mathbf P_k^- = \mathbf F_k,\mathbf P_{k-1},\mathbf F_k^\top + \mathbf Q_k

\\
\text{Kalman Gain: } \mathbf K_k = \mathbf P_k^-,\mathbf H_k^\top\bigl(\mathbf H_k,\mathbf P_k^-,\mathbf H_k^\top + \mathbf R_k\bigr)^{-1}
\end{aligned}

\\
\text{Innovation: } \tilde{\mathbf y}_k = \mathbf z_k - h\bigl(\hat{\mathbf x}_k^-\bigr)

\\
\text{Correction: } \hat{\mathbf x}_k = \hat{\mathbf x}_k^- + \mathbf K_k\tilde{\mathbf y}_k
\end{gather*}

\\
\text{Posterior covariance update: } \mathbf P_k = (\mathbf I-\mathbf K_k\mathbf H_k),\mathbf P_k^-,(\mathbf I-\mathbf K_k\mathbf H_k)^\top + \mathbf K_k\mathbf R_k\mathbf K_k^\top
$$

This set up is called "the extended Kalman Filter" (EKF)

## [3] Example: EKF For Robot Pose Estimation With GPS (Or USBL) and IMU Signal

Below is the motion model of a robot. `x` is the state vector, $p$ is the Cartesian position `x,y,z`, $v$ is the linear velocity, $q$ is the quaternion of the robot orientation. $b_a$ and $b_g$ are biases of the IMU acceleration and gyro

$$
\begin{gather*}
\begin{aligned}
\mathbf{x} =
\begin{bmatrix}
\mathbf{p} \\
\mathbf{v} \\
\mathbf{q} \\
\mathbf{b}_a \\
\mathbf{b}_g
\end{bmatrix}
\in \mathbb{R}^{16}
\end{aligned}
\end{gather*}
$$

IMU measurements are linear acceleration $\tilde{\mathbf{a}}$ in `a/m^2`, and angular velocity:

$$
\begin{gather*}
\begin{aligned}
\tilde{\mathbf{a}}, \; \tilde{\bm{\omega}} \in \mathbb{R}^3
\end{aligned}
\end{gather*}
$$

Process noise vector:

$$
\begin{gather*}
\begin{aligned}
\mathbf{w} =
\begin{bmatrix}
\mathbf{n}_a \\
\mathbf{n}_\omega \\
\mathbf{n}_{ba} \\
\mathbf{n}_{bg}
\end{bmatrix}
\sim \mathcal{N}(0, Q_c)
\end{aligned}
\end{gather*}
$$

Continuous-time dynamics:

$$
\begin{gather*}
\begin{aligned}
\dot{\mathbf{p}} &= \mathbf{v} \\
\dot{\mathbf{v}} &= R(\mathbf{q})\left(\tilde{\mathbf{a}} - \mathbf{b}_a - \mathbf{n}_a\right) + \mathbf{g} \\
\dot{\mathbf{q}} &= \tfrac{1}{2}\Gamma(\mathbf{q})\left(\tilde{\bm{\omega}} - \mathbf{b}_g - \mathbf{n}_\omega\right) \\
\dot{\mathbf{b}}_a &= \mathbf{n}_{ba} \\
\dot{\mathbf{b}}_g &= \mathbf{n}_{bg}
\end{aligned}
\end{gather*}
$$

Where $\Gamma(\mathbf{q})$ is a 4x3 matrix:

$$
\begin{gather*}
\begin{aligned}
[\mathbf{x}]_\times =
\begin{bmatrix}
0 & -x_3 & x_2 \\
x_3 & 0 & -x_1 \\
-x_2 & x_1 & 0
\end{bmatrix}
\qquad
\Gamma(\mathbf{q}) =
\begin{bmatrix}
-\mathbf{q}_v^\top \\
q_0 I_3 + [\mathbf{q}_v]_\times
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

Now, we can start developing a linearized model of the control and observation model $F_k$, and $G_k$, either using auto-diff, or deriving an analytic form by ignoring higher order terms

## [4] Why EKF Is Not The Best And Use ESKF Instead

### Reason 1 - Rotation Update Is Not Natively On the SO(3) Manifold

Traditional Extended Kalman Filters (EKF) keep the system orientation (expressed in quaternion) directly in the state vector. The update on rotation however, is still done first by

$$
\begin{gather*}
\begin{aligned}
& q = q + K(q)?
\end{aligned}
\end{gather*}
$$

followed by normalizing the quaternion. Because of the extra normalization step, errors could accumulate over time.

ESKF keeps "correction-to-current-estimate" terms as its states. The last step - the update step, rotation is instead updated by $R = R (K * \delta R)$, so no extra normalization is needed.

## Reason 2 - Loss of Significance

EKF uses raw position and orientation as state vectors. These values can vary, in the magnitudes of $10^2$ or more. The linearity of the control and observation models could vary drastically in this large range, too.

ESKF on the other hand, does not suffer this issue because its states ("correction-to-current-estimate" terms) are close to 0.

Interested in reading more about ESKF? [Please check out this article](https://ricojia.github.io/2024/03/24/robotics-full-eskf/)!
