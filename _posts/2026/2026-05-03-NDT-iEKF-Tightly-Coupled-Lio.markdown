---
layout: post
title: NDT-Tightly-Coupled-Lio
date: 2026-05-03 13:19
subtitle: iEKF
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Machine-Learning
---
# Loosely vs Tightly Coupled LIO

LiDAR-inertial odometry combines LiDAR geometry with IMU motion prediction. At a high level, there are two common ways to do this: loosely coupled fusion and tightly coupled fusion.

In a loosely coupled system, each sensor module is mostly solved independently. For example, the IMU may propagate an ESKF state, while a LiDAR odometry module independently runs ICP, NDT, LOAM-style optimization, or another scan-matching method. The LiDAR module then outputs a pose estimate, which is fused back into the filter as a high-level measurement.

In a tightly coupled system, the estimator does not wait for LiDAR to produce a complete pose solution. Instead, LiDAR residuals are inserted directly into the estimator together with the IMU prediction. The optimizer or filter jointly uses IMU state uncertainty and individual LiDAR residuals when computing the correction.

You can think of loosely coupled LIO as a design with relatively clear module boundaries, or "high cohesion, low coupling" (高内聚，低耦合): the lidar odometry front-end solves its own problem first, and the IMU or INS is fused afterward. That separation is clean from a system-design perspective, but it can make the lidar front-end more exposed to geometric degeneracy. The difference may look subtle when all sensors work well. But when LiDAR becomes geometrically weak, the two designs behave very differently.

## Gauge Freedom

In lidar-inertial odometry, gauge freedom refers to directions in the state space that are weakly observed or unobservable from the current measurements. In practice, this appears as degeneracy: different state updates produce almost the same residual error.

- If the lidar scene is feature-poor or highly repetitive, the lidar-only update can become unreliable.
- The IMU may still provide useful short-term motion propagation, but it does not directly rescue the front-end scan-matching optimization if that optimization is solved independently.
- For this reason, many systems include a degeneracy detection step. When the lidar update becomes weak, the estimator can reduce the correction, keep propagating with the inertial solution, or temporarily fall back to an INS-dominated estimate.

Consider scan matching in a long hallway, tunnel, or pipe. If the scene looks nearly identical as the sensor moves along one direction, then the lidar residuals may not distinguish motion well in that direction. For example, several poses along the corridor axis can explain the scan almost equally well.

After linearization, the optimization is usually written as

$$
H \Delta x = b, \quad H = J^T W J.
$$

Here, $J$ is the Jacobian of the residuals and $W$ is the weighting matrix. Since $H = J^T W J$, the normal matrix is positive semidefinite when $W$ is positive semidefinite. If $H$ is rank deficient or badly ill-conditioned, then there exists a nonzero vector $v$ such that

$$
H v \approx 0.
$$

That means the cost changes very little along the direction $v$, so the update is not uniquely determined by the measurements. This is the core of gauge freedom in local estimation.

Take a simplified 2D corridor. Let the state update be

$$
\Delta x = \begin{bmatrix} \Delta s \\ \Delta n \end{bmatrix},
$$

where $s$ is motion along the corridor and $n$ is motion across the corridor. If the corridor is long and visually repetitive, lidar often constrains the cross-corridor direction much better than the along-corridor direction. A toy lidar-only normal matrix might look like

$$
H_{\text{lidar}} =
\begin{bmatrix}
0.01 & 0 \\
0 & 12
\end{bmatrix}.
$$

This means:

- The cost changes only a little when we move along $s$.
- The cost changes a lot when we move along $n$.
- So the solution is numerically weak along the corridor axis.

Then the matrix is nearly singular, and updates in the $s$ direction are poorly determined.

Now suppose we add IMU constraints in a tightly coupled estimator. Over a short time interval, the IMU provides a motion prior and temporal consistency, so a simple toy contribution could look like

$$
H_{\text{imu}} =
\begin{bmatrix}
4 & 0 \\
0 & 0.5
\end{bmatrix}.
$$

The fused normal matrix becomes

$$
H_{\text{total}} = H_{\text{lidar}} + H_{\text{imu}} =
\begin{bmatrix}
4.01 & 0 \\
0 & 12.5
\end{bmatrix}.
$$

This is the key intuition:

- Lidar alone gave almost no information along $s$.
- IMU adds a short-horizon motion constraint, so the along-corridor direction is no longer almost free.
- The fused problem is better conditioned, even though global drift is still possible over long time scales.

This example is deliberately simplified. In a real LIO system, the state includes position, orientation, velocity, and IMU biases, so $H$ is much larger than $2 \times 2$. But the conditioning story is the same: the lidar block may be weak in some directions, and the IMU terms help stabilize the update.

### Why tightly coupled LIO helps

In a tightly coupled LIO system, lidar residuals and inertial constraints are fused in a single estimator. This is the main advantage:

- The IMU constrains short-term motion very strongly.
- Gravity makes roll and pitch observable in normal operation.
- Velocity and bias evolution provide additional temporal constraints between scans.
- As a result, the combined system is usually much better conditioned than lidar-only scan matching.

However, tightly coupled fusion does not remove every unobservable mode.

- Absolute global position is still unobservable without external references such as GNSS, loop closure, or landmarks with known coordinates.
- Yaw is not directly observed by the accelerometer; it is constrained only through gyroscope integration, motion, and environment geometry.
- In a highly degenerate environment, even a tightly coupled system can still become weakly constrained.

## IEKF: Iterated Error-State Kalman Filter

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
    - [Using $R^{\prime} = R\omega^{\wedge}$](https://ricojia.github.io/2017/02/22/lie-group/), $R_{wb}^{\prime} = \mathrm{Exp}\!\left((\delta\theta^{\prime})^{\wedge}\right)$
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
\left(P^{-1} + J^T\Sigma^{-1}J\right)^{-1}
\left(-J^T\Sigma^{-1}e\right)
= (P^{-1} + A)^{-1} b
$$

is equivalent to the classic ESKF/Kalman update

$$
\delta x = Ky = PJ^T\left(JPJ^T+\Sigma\right)^{-1}(-e)
$$
Proof:
$$
\begin{gather*}
\begin{aligned}
&\left(P^{-1} + J^T \Sigma^{-1} J\right) P J^T \left(J P J^T + \Sigma\right)^{-1} \\
&= P^{-1}P J^T \left(J P J^T + \Sigma\right)^{-1}
+ J^T \Sigma^{-1} J P J^T \left(J P J^T + \Sigma\right)^{-1} \\
&= J^T \left(J P J^T + \Sigma\right)^{-1}
+ J^T \Sigma^{-1} J P J^T \left(J P J^T + \Sigma\right)^{-1} \\
&= J^T \Sigma^{-1}\Sigma \left(J P J^T + \Sigma\right)^{-1}
+ J^T \Sigma^{-1} J P J^T \left(J P J^T + \Sigma\right)^{-1} \\
&= J^T \Sigma^{-1} \left(J P J^T + \Sigma\right) \left(J P J^T + \Sigma\right)^{-1} \\
&= J^T \Sigma^{-1} \\
&\Rightarrow \left(P^{-1} + J^T\Sigma^{-1}J\right)^{-1}\left(-J^T\Sigma^{-1}e\right)
= -PJ^T\left(JPJ^T+\Sigma\right)^{-1}e

\end{aligned}
\end{gather*}
$$

### Covariance Update

In classic ESKF measurement update, the simplified covariance update is

$$
P^+ = (I-KJ)\bar P
$$

Define

$$
Q_k = (\bar P^{-1}+A)^{-1}
$$

Then

$$
Q_k\left(\bar P^{-1}+A\right)=I
$$
Expanding:

$$
Q_k\bar P^{-1} + Q_kA = I
$$
And get
$$
Q_k = (I-Q_kA)\bar P
$$
Then one writes:

$$
(I-Q_kA)\bar P = Q_k = (\bar P^{-1}+A)^{-1}
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

## An example with three IMU readings, and One Lidar Observation

### Set up

The ESKF state vector is:

$$
x = [p, v, \theta, b_g, b_a, g] \in \mathbb{R}^{18}
$$

At the start of step $k$ the filter holds a nominal state $\hat{x}^-_k$ and error-state covariance $\hat{P}^-_k$, both produced by the previous correction step.

---

## Another Example: Point-to-Point ICP as Nonlinear Least Squares

Given $N$ matched pairs — source point $\mathbf{p}_i$ (current scan) and target point $\mathbf{q}_i$ (map) — the per-correspondence residual under pose $(R, \mathbf{t})$ is:

$$
\mathbf{e}_i = R\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \in \mathbb{R}^3
$$

The total cost is $\mathcal{C} = \frac{1}{2}\sum_i \lVert\mathbf{e}_i\rVert^2$.

**Jacobian** — linearize $\mathbf{e}_i$ around the current nominal pose $(\hat{R}, \hat{\mathbf{t}})$ using the error state $\delta x = [\delta \mathbf{p},\; \delta\boldsymbol{\theta}]^T \in \mathbb{R}^6$:

$$
\mathbf{e}_i \approx \underbrace{(\hat{R}\mathbf{p}_i + \hat{\mathbf{t}} - \mathbf{q}_i)}_{\mathbf{e}_i^{(0)}} + J_i\,\delta x,
\qquad
J_i = \begin{bmatrix} I_3 & -[\hat{R}\mathbf{p}_i]_\times \end{bmatrix} \in \mathbb{R}^{3\times 6}
$$

where $\times$ denotes the skew-symmetric (cross-product) matrix. The Jacobian w.r.t. translation is $I_3$; w.r.t. rotation it is $-[\hat{R}\mathbf{p}_i]_\times$ (from the first-order approximation $R \approx \hat{R}(I + [\delta\boldsymbol{\theta}]_\times)$).

**Gauss-Newton normal equations** — stacking all $N$ correspondences:

$$
\mathcal{H} = \sum_i J_i^T J_i, \qquad \mathbf{b} = \sum_i J_i^T \mathbf{e}_i^{(0)}
$$

$$
\delta x = -\mathcal{H}^{-1}\mathbf{b}
$$

**Update**: $\hat{x} \leftarrow \hat{x} \oplus \delta x$; iterate until convergence. This is pure scan-matching — no IMU prior is included.

---

### Alternative 1 — Loosely Coupled: ICP Pose as ESKF Observation

**Step 1 — IMU propagation.** Propagate the nominal state and covariance through each IMU measurement sequentially:

$$
\hat{x}^-_{k,1} = f(\hat{x}_{k-1},\, u_1)
$$

$$
\hat{x}^-_{k,2} = f(\hat{x}^-_{k,1},\, u_2)
$$

$$
\hat{x}^-_{k,3} = f(\hat{x}^-_{k,2},\, u_3) \triangleq \hat{x}^-_k
$$

Covariance propagation at each sub-step (including process noise $Q_i$):

$$
P^-_{k,1} = F_1\, P_{k-1}\, F_1^T + Q_1
$$

$$
P^-_{k,2} = F_2\, P^-_{k,1}\, F_2^T + Q_2
$$

$$
P^-_{k,3} = F_3\, P^-_{k,2}\, F_3^T + Q_3 \triangleq P^-_k
$$

**Step 2 — Run ICP independently.** Solve the ICP nonlinear least squares from the previous section using $\hat{x}^-_k$ as the initial guess. ICP returns a 6-DoF pose correction:

$$
\delta x_\text{ICP} = [\delta p_x,\; \delta p_y,\; \delta p_z,\; \delta\theta_x,\; \delta\theta_y,\; \delta\theta_z]^T
$$

**Step 3 — ESKF correction.** Treat $\delta x_\text{ICP}$ as a direct observation of the pose error state with observation Jacobian $J = [I_6 \mid 0_{6\times 12}]$ and noise covariance $R_\text{ICP}$:

$$
K = P^-_k J^T \bigl(J P^-_k J^T + R_\text{ICP}\bigr)^{-1}
$$

$$
\delta x = K\,(\delta x_\text{ICP} - J\cdot 0) = K\,\delta x_\text{ICP}
$$

$$
\hat{x}^+_k = \hat{x}^-_k \oplus \delta x, \qquad P^+_k = (I - KJ)\,P^-_k
$$

**Why is this loosely coupled and less informative?** Yes — this *is* loosely coupled: LiDAR is processed independently by ICP, which produces a pose estimate that is then fused into the filter, exactly matching the loosely-coupled definition. ICP minimizes $\sum_i \lVert\mathbf{e}_i\rVert^2$ without any knowledge of $P^-_k$. The resulting $\delta x_\text{ICP}$ is optimal for the scan-matching problem *alone*, but it discards the individual residual structure of all $N$ point pairs. When handed to the ESKF as a single synthetic observation, the filter cannot exploit those $N$ independent constraints individually — making it less informative than if the raw point residuals were incorporated directly. That is what Alternative 2 does.

---

### Alternative 2 — Iterated ESKF (IEKF)

**Is IEKF the same as Iterated ESKF?** Yes. IEKF (Iterated Extended Kalman Filter) repeatedly re-linearizes the measurement function around the current corrected estimate rather than around the propagated prior $\hat{x}^-$. When applied to the ESKF, each iteration is an ESKF correction step with updated Jacobians — so IEKF and Iterated ESKF are the same concept.

**Key difference from Alternative 1.** Instead of running a standalone ICP solver and feeding its output into the filter, IEKF embeds each point residual $\mathbf{e}_i$ *directly* as a Kalman observation. The IMU prediction covariance $P^-_k$ acts as a prior that regularizes the correction — the two sources of information are jointly optimized.

**IEKF correction step** — initialize with $\hat{x}^{(0)} = \hat{x}^-_k$ and iterate over $j = 0, 1, 2, \ldots$:

1. **Evaluate residuals** at the current iterate $\hat{x}^{(j)} = (\hat{R}^{(j)}, \hat{\mathbf{t}}^{(j)}, \ldots)$:

$$
\mathbf{r}_i^{(j)} = \hat{R}^{(j)}\mathbf{p}_i + \hat{\mathbf{t}}^{(j)} - \mathbf{q}_i
$$

$\mathbf{r}_i^{(j)}$ is called the **observation residual** (or **innovation**) evaluated at the current linearization point.

2. **Form stacked Jacobian and residual** over all $N$ pairs:

$$
J^{(j)} = \begin{bmatrix} J^{(j)}_1 \\ \vdots \\ J^{(j)}_N \end{bmatrix} \in \mathbb{R}^{3N\times 18}, \qquad
\mathbf{r}_0^{(j)} = \begin{bmatrix} \mathbf{r}_1^{(j)} \\ \vdots \\ \mathbf{r}_N^{(j)} \end{bmatrix} \in \mathbb{R}^{3N}
$$

Each row-block $J^{(j)}_i = \begin{bmatrix} I_3 & -[\hat{R}^{(j)}\mathbf{p}_i]_\times & 0_{3\times 12} \end{bmatrix}$ (zeros for velocity, biases, gravity).

3. **Kalman gain and update** — let $\tilde{x}^{(j)} = \hat{x}^{(j)} \ominus \hat{x}^-_k$ denote the accumulated drift from the prior:

$$
K^{(j)} = P^-_k \bigl(J^{(j)}\bigr)^T \Bigl(J^{(j)} P^-_k \bigl(J^{(j)}\bigr)^T + R\Bigr)^{-1}
$$

$$
\delta x^{(j)} = K^{(j)}\bigl(J^{(j)}\tilde{x}^{(j)} - \mathbf{r}_0^{(j)}\bigr)
$$

$$
\hat{x}^{(j+1)} = \hat{x}^-_k \oplus \delta x^{(j)}, \qquad P^+ = \bigl(I - K^{(j)}J^{(j)}\bigr)P^-_k
$$

At convergence, set $\hat{x}^+_k = \hat{x}^{(j+1)}$.

**Clarification on $J$ vs. $R$ in the Kalman gain.** The standard Kalman gain is:

$$
K = P J^T (J P J^T + R)^{-1}
$$

- $J$ is the **observation Jacobian** — it maps the error state $\delta x$ to the measurement space (here, 3D point residuals). It is a matrix of partial derivatives, not a covariance.
- $R$ is the **observation noise covariance** — it models how noisy each LiDAR point measurement is (e.g., range noise, surface normal uncertainty). Typically set to $\sigma^2 I_{3N}$ for isotropic point noise.
- $P^-_k$ is the **state error covariance** from IMU propagation, encoding how uncertain the predicted state is.

The term $JP^-_k J^T$ captures state-prediction uncertainty projected into measurement space; $R$ adds measurement noise on top.

**Deriving the IEKF update from the KF innovation.** In the standard KF the update is $x^+ = x^- + K(z - h(x^-))$, i.e., state-estimate plus gain times innovation. For each point pair the terms map as follows:

- **Actual measurement** $z_i = \mathbf{q}_i$ — the target/map point coordinate. This is a real sensor/map reading and is *not* zero in general.
- **Predicted measurement** $h_i(x) = R\mathbf{p}_i + \mathbf{t}$ — where the model says the source point $\mathbf{p}_i$ lands under the current pose estimate. This is a function of the state.
- **Residual** at iterate $\hat{x}^{(j)}$: $\mathbf{r}_i^{(j)} = h_i(\hat{x}^{(j)}) - z_i = \hat{R}^{(j)}\mathbf{p}_i + \hat{\mathbf{t}}^{(j)} - \mathbf{q}_i$ (predicted minus actual).

**Innovation**, linearizing $h_i(\hat{x}^-_k)$ around $\hat{x}^{(j)}$ and letting $\tilde{x}^{(j)} = \hat{x}^{(j)} \ominus \hat{x}^-_k$:

$$
z_i - h_i(\hat{x}^-_k) \approx z_i - \bigl(h_i(\hat{x}^{(j)}) + J^{(j)}_i(\hat{x}^-_k \ominus \hat{x}^{(j)})\bigr)
= -\mathbf{r}_i^{(j)} + J^{(j)}_i\,\tilde{x}^{(j)}
$$

Stacking over all $N$ pairs: innovation $= J^{(j)}\tilde{x}^{(j)} - \mathbf{r}_0^{(j)}$.

**KF update** for the error state $\delta x$ (deviation from $\hat{x}^-_k$):

$$
\delta x^{(j)} = K^{(j)}\bigl(J^{(j)}\tilde{x}^{(j)} - \mathbf{r}_0^{(j)}\bigr)
$$

## Remarks

- For observation update, FAST-LIO uses ikd-tree point-to-plane residuals instead of NDT residuals.
- The version above is a more teaching-style tightly coupled LIO view.
- FAST-LIO2 uses an incremental ikd-tree map.
