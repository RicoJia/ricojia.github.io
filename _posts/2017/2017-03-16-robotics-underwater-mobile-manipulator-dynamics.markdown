---
layout: post
title: Robotics - Spatial Momentum And Underwater Mobile Manipulator Dynamics
subtitle: Wrench, Restoring Torque, Centroidal Theorem, Thruster Allocation, Added Mass
date: '2017-03-16 13:19'
header-img: "img/post-bg-2015.jpg"
tags:
    - Robotics
    - Math
---

## Spatial Momentum

Spatial momentum $$h = [\mathbf p, \mathbf L]$$, and twist $$\mathbf v = [\mathbf v, \boldsymbol\omega]$$.

### From Momentum to Dynamics

Linear part:

$$
\mathbf p' = m \mathbf v' = m \mathbf a
$$

Angular part: $$\mathbf L = \mathbf r \times m \mathbf v$$. Since $$\mathbf r$$ is the position of the point mass w.r.t. the origin, $$\mathbf r$$ can definitely change with time, and $$\mathbf r' = \mathbf v$$. So

$$
\mathbf L' = \mathbf r' \times m\mathbf v + \mathbf r \times m\mathbf v' = \mathbf v \times m\mathbf v + \mathbf r \times m\mathbf a
$$

$$\mathbf v \times \mathbf v = 0$$, so $$\mathbf L' = \mathbf r \times m\mathbf a = \mathbf r \times \mathbf F$$, which is the definition of torque. So

$$
\boxed{h' = \text{wrench } \mathbf w = [\mathbf F, \boldsymbol\tau]}
$$

In the meantime, for a system of rigid bodies, the angular momentum about the system's center of mass $$\mathbf c$$ is (see [the centroidal model]({{ site.baseurl }}/2017/03/14/robotics-centroidal-model/)):

$$
\mathbf L = \sum_i \left[ \mathbf I_i \boldsymbol\omega_i + (\mathbf r_i - \mathbf c) \times m_i \mathbf v_i \right]
$$

$$\boldsymbol\omega_i$$ is rigid body $$i$$'s self rotation (about its own center of mass), and $$\mathbf v_i$$ is the velocity of rigid body $$i$$'s center of mass.

### Restoring Torque

1. What is restoring torque? A torque trying to rotate a body back to its equilibrium.
2. If there is $$r_x$$ and $$r_y$$ (CoB offset from CoM horizontally), is restoring torque always non-0? Yes. Even if the net force is 0 (neutrally buoyant), we get 0 torque only when buoyancy and $$(\mathbf r_B - \mathbf c)$$ are parallel, i.e., **CoB is directly above CoM**. We typically design CoB to be above CoM, so any tilt creates a torque that rotates the body back.

![](https://i.postimg.cc/xd1bqckS/Screenshot-from-2026-11-27-22-46-38.png)

## Centroidal Theorem for a Mobile Manipulator

For body $$i$$, define

$$
C_i = \begin{bmatrix} m_i I_3 & 0 \\ m_i [\mathbf r_i - \mathbf c]_\times & \mathbf I_i \end{bmatrix}
$$

where $$\mathbf I_i$$ is the inertia tensor and $$[\cdot]_\times$$ is the skew-symmetric (cross product) matrix. Then $$C_i [\mathbf v_i, \boldsymbol\omega_i] = [m_i \mathbf v_i, \ \mathbf I_i \boldsymbol\omega_i + (\mathbf r_i - \mathbf c) \times m_i \mathbf v_i]$$, which is exactly body $$i$$'s contribution to $$h$$. So for the whole system:

$$
h = C_b \mathbf v_b + C_a \mathbf v_a = h_b + h_a
$$

where $$b$$ is the base, $$a$$ is the arm, and $$\mathbf v_a$$ is the velocity of the arm w.r.t. the origin. So

$$
h' = C_b' \mathbf v_b + C_b \mathbf v_b' + C_a' \mathbf v_a + C_a \mathbf v_a'
$$

### Link Velocities

Each link's velocity can be obtained from the base twist and the arm joint velocities through the link's Jacobian:

$$
\mathbf v_i = J_i(\mathbf q) \begin{bmatrix} \mathbf v_b \\ \dot{\mathbf q} \end{bmatrix}
$$

## Wrenches

Assume you have some thrust force values $$\mathbf u = [u_1, \dots, u_n]$$, a matrix $$G$$ that maps thrust forces to a wrench (how much lifting force, how much torque), and the wrench of all externals $$\mathbf w_{ext}$$:

$$
h' = G\mathbf u + \mathbf w_{ext}
$$

How to get those: $$\mathbf w_{ext} = \mathbf w_{gravity} + \mathbf w_{buoyancy} + \mathbf w_{drag}$$ (z pointing down).

- **Gravity** is $$[0, 0, mg, 0, 0, 0]$$. The lever arm for rotation is 0, **because gravity acts on the center of mass**.
- **Buoyancy** itself is $$B = \rho_{water} V g$$. But buoyancy acts on the center of buoyancy, which is the geometric center of the displaced volume. So with $$\mathbf r_B - \mathbf c = [d_x, d_y, d_z]$$, the torque is $$(\mathbf r_B - \mathbf c) \times [0, 0, -B] = [-d_y B, \ d_x B, \ 0]$$.
    - Note, a neutrally buoyant body can still experience restoring torque.
    - Note, we get roll and pitch torque, but not yaw torque.
- **Drag** is $$F = \frac{1}{2} \rho_{water} C_d A v^2$$
    - $$A$$ is the projected cross-section area
    - $$v = v_{boat} - v_{water}$$, the relative speed
    - Drag acts on the center of drag, in the opposite direction to the velocity

![](https://i.postimg.cc/zv32qzND/download-(4).jpg)
*Center of buoyancy*

### Thrusters

For thrusts, there is an added mass $$m_a$$, which is a virtual mass due to the water that also gets accelerated by the vehicle:

$$
(m + m_a) \mathbf a = \mathbf F_{thrust}
$$

**$$G$$ is from geometry.** Thruster $$i$$ produces force $$u_i \mathbf d_i$$, where $$\mathbf d_i$$ is the direction of force. Its torque is $$(\mathbf r_i - \mathbf c) \times (u_i \mathbf d_i)$$. So column $$i$$ of $$G$$ is

$$
G_i = \begin{bmatrix} \mathbf d_i \\ (\mathbf r_i - \mathbf c) \times \mathbf d_i \end{bmatrix}
$$

Normally, thrust has a second-order relationship with the command $$s_i$$: $$u_i = k_i \, s_i^2 \, \text{sign}(s_i)$$.

![](https://i.postimg.cc/GhdJP3Vm/download-(5).jpg)
*Torque of thrusts*

## Base Acceleration

Putting it together:

$$
\mathbf v_b' = C_b^{-1} \left( G\mathbf u + \mathbf w_{ext} - C_b' \mathbf v_b - C_a \mathbf v_a' - C_a' \mathbf v_a \right)
$$

From the URDF, you know the mass matrices $$C_b, C_a$$ and their first-order derivatives.

## Added Mass

We need the wrench of added mass: $$m_a$$ is here because when the vehicle moves, some water is being accelerated too. In general, it's a matrix rather than a scalar, since the vehicle accelerates different amounts of water in different directions.

## Unknown Dynamics

- Drag
- Added mass
- Thruster interactions?
- Unmodeled fluid effects
- Vehicle-arm fluid coupling
