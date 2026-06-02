---
layout: post
title: NDT-Tightly-Coupled-Lio
date: 2026-05-03 13:19
subtitle: ""
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---
## Introduction

IEKF: Iterated Error-State Kalman Filter

IMU gives a drifting prior, and NDT gives a geometric pull-back-to-map constraint. IEKF fuses them by solving a small weighted least-squares problem on the error state, then relinearizing and solving again.

Error state:

$$
\delta x = [\delta p, \delta v, \delta \theta, \delta b_g, \delta b_a, \delta g]
$$

Nominal state:

$$
x = [p, v, \theta, b_g, b_a, g]
$$

## Prediction (IMU)

- Measurement model (Biases represent how much the measured value drifts from the true value, e.g. $\omega_m = \omega + b_g$):
 	- $\omega_m = \omega + b_g + \epsilon_\omega$
 	- $a_m = R_{bw}(a-g) + b_a + \epsilon_a$
- So from measurements $a_m, \omega_m$:
 	- $\omega = \omega_m - b_g$, $a = R_{wb}(a_m-b_a)+g$
 	- $p' = p + v\Delta t + \frac{1}{2}a\Delta t^2$
 	- $v' = v + a\Delta t$
 	- $R_{wb}' = R_{wb}R_{bb'} = R_{wb}\exp(\omega\Delta t)$
 	- $\theta = \log(R_{wb}')$
- Covariance propagation:
 	- Linearize $\delta\theta'$:
  		- $R_{wb}' = R_{wb}R_{bb} = R_{wb}\mathrm{Exp}(\delta\theta)$
  		- [Using $R' = R\omega^\land$](https://ricojia.github.io/2017/02/22/lie-group/), $R_{wb}' = \mathrm{Exp}[\delta\theta](\delta\theta')^\land$
  		- $(\delta\theta)' \approx -(\tilde{\omega}-b_g)^\land\delta\theta - \delta b_g - \eta_g$
 	- Linearize $\delta v'$:
  		- $\delta v' = -R(\tilde a - b_a)^\land\delta\theta - R\delta b_a - \eta_a + \delta g$
 	- By definition:
  		- $\delta p' = \delta v$
  		- $\delta b_g' = \eta_{bg}$
  		- $\delta b_a' = \eta_{ba}$
 	- Organize into $F$:

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
0 & I & -R(\tilde{a}-b_a)^{\land}\Delta t & 0 & -R \Delta t & I\Delta t \\
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

- With IMU covariance matrix $Q$, the prediction covariance update is:

$$
\begin{gather*}
\begin{aligned}
& P_{k+1}* = F P_{k} F^T + Q
\end{aligned}
\end{gather*}
$$

## Observation (NDT)

Different papers use different letters ($H, V, r$). For consistency here, I use:

| Other EKF notation                | Our NDT / optimization notation        |
| --------------------------------- | -------------------------------------- |
| $H$, or $C$, measurement Jacobian | $J$, combined / stacked Jacobian       |
| $V$, measurement covariance       | $\Sigma$, stacked NDT covariance       |
| $V^{-1}$                          | $\Sigma^{-1}$, stacked NDT information |
| $r$                               | $\text{error}$, stacked NDT residual   |
| $H^\top V^{-1}H$                  | $J^\top \Sigma^{-1}J$                  |
| $-H^\top V^{-1}r$                 | $-J^\top \Sigma^{-1}\text{error}$      |

Here is how to incorporate NDT into the ESKF framework

- Define point residual:
  - $e = R_{wb}x + t - q_s$
  - Total residual term: $\text{error}_i = e_i^T\Sigma^{-1}e_i$
- Jacobian:
  - $\frac{\partial e_i}{\partial R} = -Rp_t^\land$
  - $\frac{\partial e_i}{\partial t} = I$
- For vanilla NDT, we solve:

$$
\arg\min_{\delta x}\sum_i\|e_i + J\delta x_i\|_{\Sigma^{-1}}^2
$$

- Define:
  - $A = \sum_i J_i^T\Sigma^{-1}J_i = J^T\Sigma^{-1}J$
  - $b = -\sum_i J_i^T\Sigma^{-1}e_i$
  - $\chi^2 = \sum_i e_i^T\Sigma^{-1}e_i$
  - In vanilla NDT: $\delta x = A^{-1}b$

- In ESKF language, observation $H$ is equivalent to the Hessian approximation here, and observation $z$ aligns with the linearized residual term (you can also view it as corresponding to the $b$ term after linearization):

$$
\begin{gather*}
\begin{aligned}
& z = H\cdot\delta x + v
\\ &
\Rightarrow
\\ &
H = \frac{\partial h}{\partial x}\frac{\partial x}{\partial\delta x}
\end{aligned}
\end{gather*}
$$

- **This is the core iEKF idea: optimize both the IMU prediction consistency term and the NDT observation term together.**

$$
\begin{gather*}
\delta x^*
=
\arg\min_{\delta x}
\left(
\|\delta x\|_{\bar P^{-1}}^2
+
\|e + J\delta x\|_{\Sigma^{-1}}^2
\right)
\\
\delta x^* = \arg\min_{\delta x}(\delta x^TP^{-1}\delta x + \|e + J\delta x\|_{\Sigma^{-1}}^2)
\end{gather*}
$$

- Taking derivative and setting to zero:

$$
\begin{gather*}
\bar P^{-1}\delta x
+
J^T\Sigma^{-1}e
+
J^T\Sigma^{-1}J\delta x
=
0
\\
\left(
\bar P^{-1}
+
J^T\Sigma^{-1}J
\right)
\delta x
=
-
J^T\Sigma^{-1}e
\\
\left(
\bar P^{-1}
+
A
\right)
\delta x
=
b
\\
\delta x
=
\left(
\bar P^{-1}
+
A
\right)^{-1}
b
\end{gather*}
$$

This observation update is equivalent to the classic ESKF update:

$$
\delta x = Ky
$$

where

$$
K
=
\bar P J^T
\left(
J\bar P J^T+\Sigma
\right)^{-1}
$$

and:

$$
y=-e
$$

### IEKF Update = Classic Kalman Gain

We want to show the information-form update

$$  
\delta x =

\left(  
P^{-1}  
+  
J^T\Sigma^{-1}J  
\right)^{-1}  
\left(  
-J^T\Sigma^{-1}e  
\right)  
= (P^{-1} + A)^{-1} b
$$

is equivalent to the classic ESKF/Kalman update

$$  
\delta x = Ky  =
PJ^T  
\left(  
JPJ^T+\Sigma  
\right)^{-1} (-e)
$$
Proof:
$$
\begin{gather*}
\begin{aligned}
&
\left(P^{-1} + J^T \Sigma^{-1} J\right)
P J^T
\left(J P J^T + \Sigma\right)^{-1}
\\
&=
P^{-1}P J^T
\left(J P J^T + \Sigma\right)^{-1}
+
J^T \Sigma^{-1} J P J^T
\left(J P J^T + \Sigma\right)^{-1}
\\
&=
J^T
\left(J P J^T + \Sigma\right)^{-1}
+
J^T \Sigma^{-1} J P J^T
\left(J P J^T + \Sigma\right)^{-1}
\\
&=
J^T \Sigma^{-1}\Sigma
\left(J P J^T + \Sigma\right)^{-1}
+
J^T \Sigma^{-1} J P J^T
\left(J P J^T + \Sigma\right)^{-1}
\\
&=
J^T \Sigma^{-1}
\left(J P J^T + \Sigma\right)
\left(J P J^T + \Sigma\right)^{-1}
\\
&=
J^T \Sigma^{-1}

\\ & \Rightarrow
\left(
P^{-1}
+
J^T\Sigma^{-1}J
\right)^{-1}
\left(
-J^T\Sigma^{-1}e
\right)
=
-
PJ^T
\left(
JPJ^T+\Sigma
\right)^{-1}
e

\end{aligned}
\end{gather*}
$$

### Covariance Update

In classic ESKF measurement update, the simplified covariance update is

$$  
P^+ =

(I-KJ)\bar P,  
$$

Define

 $$  
Q_k =

(\bar P^{-1}+A)^{-1}.  
$$

Then

$$  
Q_k  
\left(  
\bar P^{-1}+A  
\right)

= I.  
$$
Expanding:

 $$  
Q_k\bar P^{-1}  
+  
Q_kA

=I.  
$$
And get
$$  
Q_k =

(I-Q_kA)\bar P.  
$$
Then one writes:

 $$  
(I-Q_kA)\bar P

= Q_k

= (\bar P^{-1}+A)^{-1}.  
$$

### Update

Just like ESKF:

$$
x = x \oplus \delta x
$$

NDT provides a linearized observation to the ESKF. For a fixed linearization point, each point residual is approximated as:  `-e_i ≈ J_i δx + v_i`. Stacking all points gives:    -e ≈ J δx + v. Thus the NDT update can be written as a standard ESKF measurement update.

The "iterated" part simply means that after injecting δx, we recompute e and J at the updated nominal pose and perform another ESKF-style update. NDT performs better with the re-linearization.

```cpp
for (int iter = 0; iter < options_.num_iterations_; ++iter) {
    // 1. Recompute NDT residual/Jacobian at the current nominal pose
    obs(GetNominalSE3(), HTVH, HTVr);

    // 2. Project covariance
    Mat18T J = Mat18T::Identity();
    J.template block<3, 3>(6, 6) =
        Mat3T::Identity() - 0.5 * SO3::hat((R_.inverse() * start_R).log());
    Pk = J * cov_ * J.transpose();

    // 3. Solve IEKF correction
    Qk = (Pk.inverse() + HTVH).inverse();
    dx_ = Qk * HTVr;

    // 4. Inject dx into nominal state
    Update();

    // 5. Stop if correction is small
    if (dx_.norm() < options_.quit_eps_) {
        break;
    }
}
```

## Remarks

- For observation update, FAST-LIO uses ikd-tree point-to-plane residuals instead of NDT residuals.
- The version above is a more teaching-style tightly coupled LIO view.
- FAST-LIO2 uses an incremental ikd-tree map.
