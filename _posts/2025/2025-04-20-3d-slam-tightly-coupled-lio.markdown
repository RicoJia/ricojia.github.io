---
layout: post
title: "[3D-SLAM 6] Tightly Coupled Lio"
date: 2025-04-20 13:19
subtitle:
comments: true
header-img: img/post-bg-unix-linux.jpg
tags:
  - Robotics
---

## Introduction

If the state estimation process jointly optimizes the states related to multiple sensors rather than treating each module independently, the system is considered **tightly coupled**. For example, an IMU-based system is tightly coupled when it estimates the full inertial state—including position, velocity, orientation, IMU biases, and noise parameters—within a unified optimization or filtering framework. A system tightly couples LiDAR when its optimizer directly incorporates **LiDAR scan-matching residuals** into the state estimation. Similarly, if the system uses **image feature observations and their reprojection errors** in the same optimizer loop, it becomes a tightly coupled visual system. For RTK-GNSS, tightly coupled estimation may incorporate **satellite observations, carrier-phase measurements, and satellite visibility information** directly into the estimator.

In contrast, a **loosely coupled system** processes each sensor independently and fuses only the **high-level outputs** of each module, such as pose estimates, velocity estimates, or position measurements.

In practice, **when all sensor modules function properly, the performance difference between tightly coupled and loosely coupled systems is often small**. However, when one sensor becomes **unavailable or unreliable, their behaviors diverge significantly**. For instance, if GNSS signals are lost, an INS-only solution will quickly diverge in position and velocity due to accumulated drift. Similarly, pure LiDAR or visual odometry systems can **degenerate in environments with poor geometric constraints**, such as long straight hallways or textureless scenes.

In loosely coupled systems, when a sensor module produces invalid outputs or fails entirely, **additional logic is required to detect and reject these faulty measurements**. For example, in a degenerate environment, LiDAR scan matching **may have multiple plausible solutions due to insufficient constraints**. Selecting an incorrect solution may cause **significant drift** in the overall system estimate.

In contrast, tightly coupled systems allow measurements from other sensors to **provide additional constraints** when one sensor becomes unreliable. For example, in LiDAR scan matching, the **gauge freedom** caused by geometric degeneracy can be partially constrained by inertial measurements from the IMU. As a result, the system can maintain a more stable estimate and typically drifts more slowly in challenging environments.

## An example with three IMU readings, and One Lidar Observation

### Set up

The ESKF state vector is:

$$
x = [p, v, \theta, b_g, b_a, g] \in \mathbb{R}^{18}
$$

At the start of step $k$ the filter holds a nominal state $\hat{x}^-_k$ and error-state covariance $\hat{P}^-_k$, both produced by the previous correction step.

---

### Refresher: Point-to-Point ICP as Nonlinear Least Squares

Given $N$ matched pairs — source point $\mathbf{p}_i$ (current scan) and target point $\mathbf{q}_i$ (map) — the per-correspondence residual under pose $(R, \mathbf{t})$ is:

$$
\mathbf{e}_i = R\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i \in \mathbb{R}^3
$$

The total cost is $\mathcal{C} = \frac{1}{2}\sum_i \|\mathbf{e}_i\|^2$.

**Jacobian** — linearize $\mathbf{e}_i$ around the current nominal pose $(\hat{R}, \hat{\mathbf{t}})$ using the error state $\delta x = [\delta \mathbf{p},\; \delta\boldsymbol{\theta}]^T \in \mathbb{R}^6$:

$$
\mathbf{e}_i \approx \underbrace{(\hat{R}\mathbf{p}_i + \hat{\mathbf{t}} - \mathbf{q}_i)}_{\mathbf{e}_i^{(0)}} + J_i\,\delta x,
\qquad
J_i = \begin{bmatrix} I_3 & -[\hat{R}\mathbf{p}_i]_\times \end{bmatrix} \in \mathbb{R}^{3\times 6}
$$

where $[\cdot]_\times$ denotes the skew-symmetric (cross-product) matrix. The Jacobian w.r.t. translation is $I_3$; w.r.t. rotation it is $-[\hat{R}\mathbf{p}_i]_\times$ (from the first-order approximation $R \approx \hat{R}(I + [\delta\boldsymbol{\theta}]_\times)$).

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

**Why is this loosely coupled and less informative?** Yes — this *is* loosely coupled: LiDAR is processed independently by ICP, which produces a pose estimate that is then fused into the filter, exactly matching the loosely-coupled definition. ICP minimizes $\sum_i \|\mathbf{e}_i\|^2$ without any knowledge of $P^-_k$. The resulting $\delta x_\text{ICP}$ is optimal for the scan-matching problem *alone*, but it discards the individual residual structure of all $N$ point pairs. When handed to the ESKF as a single synthetic observation, the filter cannot exploit those $N$ independent constraints individually — making it less informative than if the raw point residuals were incorporated directly. That is what Alternative 2 does.

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
