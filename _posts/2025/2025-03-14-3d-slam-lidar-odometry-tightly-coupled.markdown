---
layout: post
title: Robotics - [3D SLAM - 5] Tightly Coupled Lidar Inertial Odometry
date: 2025-03-14 13:19:00
subtitle: Gauge Freedom, Corridor Degeneracy, and FastLIO
header-img: img/post-bg-o.jpg
tags:
  - Robotics
  - SLAM
comments: true
---
## Gauge Freedom

In lidar-inertial odometry, gauge freedom refers to directions in the state space that are weakly observed or unobservable from the current measurements. In practice, this appears as degeneracy: different state updates produce almost the same residual error.

You can think of loosely coupled LIO as a design with relatively clear module boundaries, or "high cohesion, low coupling" (高内聚，低耦合): the lidar odometry front-end solves its own problem first, and the IMU or INS is fused afterward. That separation is clean from a system-design perspective, but it can make the lidar front-end more exposed to geometric degeneracy.

### Why it happens

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

### Gauge freedom in loosely coupled LIO

In a loosely coupled LIO system, the lidar odometry front-end often estimates pose first, and the IMU information is fused afterward. This separation makes the lidar estimator more vulnerable to geometric degeneracy.

- If the lidar scene is feature-poor or highly repetitive, the lidar-only update can become unreliable.
- The IMU may still provide useful short-term motion propagation, but it does not directly rescue the front-end scan-matching optimization if that optimization is solved independently.
- For this reason, many systems include a degeneracy detection step. When the lidar update becomes weak, the estimator can reduce the correction, keep propagating with the inertial solution, or temporarily fall back to an INS-dominated estimate.

### Corridor example

Take a simplified 2D corridor. Let the state update be

$$
\Delta x = \begin{bmatrix} \Delta s \\ \Delta n \end{bmatrix},
$$

where $s$ is motion along the corridor and $n$ is motion across the corridor.

An intuition sketch looks like this:

```text
wall                    wall
|                        |
|   sensor -> -> ->      |
|                        |
|________________________|

s: along the corridor axis
n: across the corridor axis
```

If the corridor is long and visually repetitive, lidar often constrains the cross-corridor direction much better than the along-corridor direction. A toy lidar-only normal matrix might look like

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

So the right statement is not that the IMU magically makes $H$ positive definite in all cases. The more accurate statement is that IMU constraints usually reduce degeneracy and improve numerical stability, especially over short time intervals.

