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
## 1 - Loosely vs Tightly Coupled LIO

LiDAR-inertial odometry combines LiDAR geometry with IMU motion prediction. At a high level, there are two common ways to do this: loosely coupled fusion and tightly coupled fusion.

In a loosely coupled system, each sensor module is mostly solved independently. For example, the IMU may propagate an ESKF state, while a LiDAR odometry module independently runs ICP, NDT, LOAM-style optimization, or another scan-matching method. The LiDAR module then outputs a pose estimate, which is fused back into the filter as a high-level measurement.

In a tightly coupled system, the estimator does not wait for LiDAR to produce a complete pose solution. Instead, LiDAR residuals are inserted directly into the estimator together with the IMU prediction. The optimizer or filter jointly uses IMU state uncertainty and individual LiDAR residuals when computing the correction.

You can think of loosely coupled LIO as a design with relatively clear module boundaries, or "high cohesion, low coupling" (高内聚，低耦合): the lidar odometry front-end solves its own problem first, and the IMU or INS is fused afterward. That separation is clean from a system-design perspective, but it can make the lidar front-end more exposed to geometric degeneracy. The difference may look subtle when all sensors work well. But when LiDAR becomes geometrically weak, the two designs behave very differently.

---

## 2 - IEKF: Iterated Error-State Kalman Filter

IMU gives a drifting prior, and NDT gives a geometric pull-back-to-map constraint. IEKF fuses them by solving a small weighted least-squares problem on the error state, then relinearizing and solving again.

Error state:

$$
\delta x = [\delta p, \delta v, \delta \theta, \delta b_g, \delta b_a, \delta g]
$$

Nominal state:

$$
x = [p, v, \theta, b_g, b_a, g]
$$

### 3 - Prediction with IMU

### 3 - 1 Contiunous State Transition

The IMU measurements are

$$  
\omega_m=\omega+b_g+n_g,  
$$
$$  
a_m=R_{bw}(a_w-g)+b_a+n_a.  
$$

Therefore, using the current bias estimates,

$$  
\hat\omega=\omega_m-\hat b_g,  
$$
$$  
\hat a_w =

R_{wb}(a_m-\hat b_a)+\hat g.  
$$

The nominal state is propagated as
$$  
p_{k+1} =

p_k+v_k\Delta t+\frac12\hat a_w\Delta t^2,  
$$
$$  
v_{k+1} =

v_k+\hat a_w\Delta t,  
$$
$$  
R_{k+1} =

R_k\operatorname{Exp}  
\left(  
[\hat\omega]_\times\Delta t  
\right).  
$$
The biases and gravity are kept constant during nominal propagation:

$$  
b_{g,k+1}=b_{g,k},  
\qquad  
b_{a,k+1}=b_{a,k},  
\qquad  
g_{k+1}=g_k.  
$$

Their uncertainty still grows through the process noise.

Use the error state

$$  
\delta x=  
\begin{bmatrix}  
\delta p&  
\delta v&  
\delta\theta&  
\delta b_g&  
\delta b_a&  
\delta g  
\end{bmatrix}^{\top}.  
$$

For a right rotation error, the continuous-time linearized dynamics are approximately

$$  
\delta\dot p=\delta v,  
$$
$$  
\delta\dot v =

-R[a_m-\hat b_a]_\times\delta\theta  
-R\delta b_a  
+\delta g  
-Rn_a,  
$$
$$  
\delta\dot\theta =

-[\omega_m-\hat b_g]_\times\delta\theta  
-\delta b_g  
-n_g,  
$$
$$  
\delta\dot b_g=n_{bg},  
\qquad  
\delta\dot b_a=n_{ba},  
\qquad  
\delta\dot g=0.  
$$

### 3 - 2 Discrete State Transition

**After discretization,**

$$
\delta x_{k+1} =

F_k\delta x_k+G_dw_k.  
$$

Using a first-order discretization,

$$  
F_k  
\approx  
\begin{bmatrix}  
I&I\Delta t&0&0&0&0\\  
0&I&-R[\hat a_b]_\times\Delta t&0&-R\Delta t&I\Delta t\\  
0&0&\operatorname{Exp}(-[\hat\omega]_\times\Delta t)&-I\Delta t&0&0\\  
0&0&0&I&0&0\\  
0&0&0&0&I&0\\  
0&0&0&0&0&I  
\end{bmatrix},  
$$

where

$$  
\hat a_b=a_m-\hat b_a.  
$$

The ($-I\Delta t$) gyro-bias block is a first-order approximation. A more accurate discretization uses the (SO(3)) right Jacobian.

### 3 - 3 Covariance propagation

The predicted covariance is

$$  
\boxed{  
P_{k+1}^{-} =

F_kP_k^{+} F_k^\top  
+  
G_dQ_wG_d^\top  
}  
$$

where ($Q_w$) contains the gyro noise, accelerometer noise, and bias random-walk covariances. If the noise mapping has already been included, define

$$  
Q_d=G_dQ_wG_d^\top  
$$

and write

$$  
\boxed{  
P_{k+1}^{-} =

F_kP_k^{+} F_k^\top+Q_d.  
}  
$$

The identity blocks for ($b_g$), ($b_a$), and ($g$) mean that these errors are constant during deterministic propagation. They can still be corrected during the LiDAR measurement update.

## 4 - Observation Update with NDT

Different papers use different symbols. Here, we use:

| EKF notation      | NDT notation          | Meaning                        |
| ----------------- | --------------------- | ------------------------------ |
| H or C            | J                     | Stacked measurement Jacobian   |
| V or R            | $\Sigma$              | Stacked measurement covariance |
| r                 | e                     | Stacked residual vector        |
| $H^\top V^{-1}H$  | $J^\top\Sigma^{-1}J$  | Measurement information matrix |
| $-H^\top V^{-1}r$ | $-J^\top\Sigma^{-1}e$ | Residual gradient term         |

### 4 - 1 NDT residual

For a source point $p_i$, let $\mu_i$ and $\Sigma_i$ be the mean and covariance of its matched NDT voxel.

The transformed point is

$$  
p_i^w = Rp_i+t.  
$$

The residual is

$$  
e_i = Rp_i+t-\mu_i.  
$$

Its weighted cost is

$$  
\chi_i^2=e_i^\top\Sigma_i^{-1}e_i.  
$$

Here, ($e_i$) is a vector. The value ($\chi_i^2$) is a scalar cost.

### 4 - 2 Residual Jacobian

Assuming a right rotation perturbation,

$$  
R_{\text{new}}=R\operatorname{Exp}(\delta\theta),  
$$

the linearized residual is

$$  
e_i(\delta x)\approx e_i+J_i\delta x.  
$$

The position and rotation blocks are

$$  
\frac{\partial e_i}{\partial \delta p}=I,  
\qquad  
\frac{\partial e_i}{\partial \delta\theta}

-R[p_i]_\times.  
$$

For the state ordering

$$  
\delta x=  
\begin{bmatrix}  
\delta p&  
\delta v&  
\delta\theta&  
\delta b_g&  
\delta b_a&  
\delta g  
\end{bmatrix}^\top,  
$$

the full Jacobian is

$$  
J_i=  
\begin{bmatrix}  
I&  
0&  
-R[p_i]_\times&  
0&  
0&  
0  
\end{bmatrix}.  
$$

---

### 4.3 Iterated State Update and One-shot Covariance Update

Let

- $\hat x^-$ be the propagated IMU state;

- $\bar P$ be its covariance;

- $\hat x_k$ be the nominal state at iteration $k$.

At each iteration, NDT computes the residual and Jacobian at the current state:

$$  
-e_k  
\approx  
J_k \delta x_k +v.  
$$

The IEKF then:

1. Recomputes NDT correspondences, residuals, and Jacobians.

2. Solves for the correction $\delta x_k$.

3. Injects the correction into the nominal state.

4. Repeats until the correction is small.

The same propagated IMU prior is used throughout these iterations because they all process the same LiDAR scan.

#### 4-3-1 First iteration

At the first iteration,

$$  
\hat x^0=\hat x^-.  
$$

Therefore, the prior error has zero mean, and the optimization is

$$  
\delta x^* =

\arg\min_{\delta x}  
\left[  
\lVert\delta x\rVert_{\bar P^{-1}}^2  
+  
\lVert e+J\delta x\rVert_{\Sigma^{-1}}^2  
\right].  
$$

This gives

$$  
\boxed{  
\delta x =

\left(  
\bar P^{-1}  
+  
J^\top\Sigma^{-1}J  
\right)^{-1}  
\left(  
-J^\top\Sigma^{-1}e  
\right).  
}  
$$

Define

$$  
A=J^\top\Sigma^{-1}J,  
\qquad  
b=-J^\top\Sigma^{-1}e.  
$$

Then

$$  
\boxed{  
\delta x=(\bar P^{-1}+A)^{-1}b.  
}  
$$

#### 4-3-2 Later iterations

After the first correction,

$$  
\hat x_k\neq\hat x^-.  
$$

Define the displacement from the propagated state as

$$  
d_k =

\hat x_k\boxminus\hat x^-.  
$$

Let $T_k$ map an error in the current tangent space into the tangent space of the propagated state:

$$  
(\hat x_k\boxplus\delta x)  
\boxminus\hat x^-  
\approx  
d_k+T_k\delta x.  
$$

The correct objective is therefore

$$  
\boxed{  
\delta x_{k} ^* =

\arg\min_{\delta x}  
\left[  
\left|  
d_k+T_k\delta x  
\right|_{\bar P^{-1}}^2  
+  
\left|  
e_k+J_k\delta x  
\right|_{\Sigma^{-1}}^2  
\right].  
}  
$$

The normal equation is

$$  
\begin{aligned}  
\Big[  
(T_k)^\top\bar P^{-1}T_k  
+  
(J_k)^\top\Sigma^{-1}J_k  
\Big]\delta x_k  
={}&  
-(T_k)^\top\bar P^{-1}d_k\  
&-(J_k)^\top\Sigma^{-1}e_k.  
\end{aligned}  
$$

The first term on the right keeps the current iterate consistent with the original IMU prediction. It becomes zero only during the first iteration.

FAST-LIO2 expresses the same idea by transporting the prior covariance into the current tangent space and including the displacement between the current iterate and the propagated state in every correction. ([ar5iv](https://ar5iv.labs.arxiv.org/html/2107.06829 "[2107.06829] FAST-LIO2: Fast Direct LiDAR-inertial Odometry"))

Note, $T_k$ is a Jacobian that relates the tangent space of the current nominal state, $\hat x_k$ and the tangent space of the propagated IMU state $\hat x^-$. For position, velocity, biases, and gravity, the state change is additive. so the projection of the state change is identity. Rotation is a bit more complicated, I haven't fully gotten that yet, but this looks like a right Jacobian using BCH:

$$  
\boxed{  
T_k =

\operatorname{diag}  
\left(  
I,,  
I,,  
J_r^{-1}(\phi_k),,  
I,,  
I,,  
I  
\right).  
}  
$$

### 4-3-3 Equivalent transported-prior form

Define

$$  
P_k =

(T_k)^{-1}  
\bar P  
(T_k)^{-\top},  
$$

and the prior mean in the current tangent space:

$$  
\mu_k =

-(T_k)^{-1}d_k.  
$$

Then

$$  

\boxed{  
\delta x_k =

\left[  
(P_k)^{-1}+A_k  
\right]^{-1}  
\left[  
(P_k)^{-1}\mu_k+b_k  
\right].  
}  
$$

During the first iteration,

$$  
T_0=I,  
\qquad  
d_0=0,  
\qquad  
\mu_0=0,  
$$

so this reduces to the simpler equation derived earlier.

---

### 4.4 Final Covariance Update

The covariance should be updated using the Jacobian at the converged state because **covariance describes the local uncertainty around the final estimate**. In a nonlinear measurement model, the residual and Jacobian depend on the current pose:

$$  
e_k=e(x_k),  
\qquad  
J_k=\left.\frac{\partial e}{\partial \delta x}\right|_{x_k}.  
$$

During the IEKF iterations, $x_k$, the NDT correspondences, and J_k may all change. A Jacobian computed at an early iteration describes the measurement geometry around an intermediate—and possibly inaccurate—state. After convergence, the final Jacobian $J$ provides the best local linear approximation around the accepted state $x$. Therefore, the posterior covariance is computed once as

Let ($P_k$) be the propagated covariance transported into the tangent space of the converged state. Define

$$  
A_k =

(J_k)^\top  
\Sigma^{-1}  
J_k.  
$$

The posterior covariance is

$$  

\boxed{  
P^+ =

\left[  
(P_k)^{-1}+A_k  
\right]^{-1}.  
}  
$$

Equivalently (Rico: you can plugin k and realize that below is the same as the above),

$$  
K_k =

P_k(J_k)^\top  
\left(  
J_k P_k(J_k)^\top+\Sigma  
\right)^{-1},  
$$

and

$$  

\boxed{  
P^+ =

(I-K_k J_k)P_k.  
}  
$$

FAST-LIO2 similarly iterates the state until convergence and then computes the final covariance from the final gain, measurement Jacobian, and transported prior covariance. ([ar5iv](https://ar5iv.labs.arxiv.org/html/2107.06829 "[2107.06829] FAST-LIO2: Fast Direct LiDAR-inertial Odometry"))

If the covariance is not already expressed in the tangent space of the final injected state, apply the required reset or tangent-space transport afterward.

---

## 5 - Where TIghtly Coupeld LIO Shines - Gauge Freedom

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

---

### 5 - 1 Why tightly coupled LIO helps

In a tightly coupled LIO system, lidar residuals and inertial constraints are fused in a single estimator. This is the main advantage:

- The IMU constrains short-term motion very strongly.
- Gravity makes roll and pitch observable in normal operation.
- Velocity and bias evolution provide additional temporal constraints between scans.
- As a result, the combined system is usually much better conditioned than lidar-only scan matching.

However, tightly coupled fusion does not remove every unobservable mode.

- Absolute global position is still unobservable without external references such as GNSS, loop closure, or landmarks with known coordinates.
- Yaw is not directly observed by the accelerometer; it is constrained only through gyroscope integration, motion, and environment geometry.
- In a highly degenerate environment, even a tightly coupled system can still become weakly constrained.
