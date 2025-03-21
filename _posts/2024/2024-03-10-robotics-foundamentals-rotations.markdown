---
layout: post
title: Robotics Fundamentals - Rotations
date: '2024-03-10 13:19'
subtitle: Representation of Rotations, Gimbal Lock
comments: true
tags:
    - Robotics
---

## Representations of Rotation

A rotation, can be respresented as $so(3)$ (Lie Algebra of Special Orthogonal Group), or $SO(3)$, (Special Orthogonal Group) and rotation vector.

**Representation 1** A rotation vector is $s = \theta [s_x, s_y, s_z] = [\omega_x, \omega_y, \omega_z]$, where:

$$
\begin{gather*}
\theta = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}
\end{gather*}
$$

$[s_x, s_y, s_z]$ here is the axis of rotation, which is a unit vector.

**Representation 2** Then we can write this rotation vector in the form of $so(3)$. It's also called "skew symmetric matrix" of a rotation axis (notice how the matrix diagonal serve as the axis of symmetry and sign?)

$$
\begin{gather*}
\hat{\omega} = \begin{pmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{pmatrix}
\end{gather*}
$$

**Representation 3** [SO(3)](../2017/2017-02-19-lie-group.markdown)


### Order is important

Different orders of Rotation about x, y, z axis could yield different final orientations. Try it yourself: first, rotate a random object you see now by z axis by +90 degrees followed by rotating about x axis by +90 degrees. Then, try rotating about x axis followed by about the z axis. A side note is that in robotics, people follow the **right hand convention** for frames. That is: x forward, y left, z up

So, **it's important to specify the order or rotation, if a single rotation can be decomposed into rotations about x,y,z axes.**. Also, there are rotations about **fixed axes**, and **Euler angles**. Fixed axes refer to the axes of a fixed world frame, and euler angles refer to the X,Y,Z in the body frame. In either case, x,y,z are also known as "roll-pitch-yaw".

In the robotics community:

- Use world frame fixed axes. [ROS uses the X-Y-Z order](https://www.ros.org/reps/rep-0103.html), so there is no ambiguity on order. There is no gimbal lock, either.

- Euler angles can represent any orientation. There are common orders such as Z-X-Y, X-Y-Z, etc. There are 24 valid combinations. Below (from Wikipedia) is an illustration of Z-X-Z'

<p align="center">
<img src="https://github.com/ChengeYang/Probabilistic-Robotics-Algorithms/assets/39393023/c15bc499-b0af-49a4-b773-17354dab8d4e" height="400" width="width"/>
</p>

### Multiple Rotations Leads To One Rotation

The final rotation of 3 rotations about fixed axes is
$$
\begin{gather*}
R = R_z(\theta_z) R_y(\theta_y) R_x(\theta_x)
\end{gather*}
$$

Then we can get $\theta$ and rotation axis $u=[u_x, u_y, u_z]$

$$
\begin{gather*}
\theta = \cos^{-1} \left( \frac{\text{trace}(R) - 1}{2} \right)
\\
\mathbf{u} = \frac{1}{2 \sin \theta} \begin{pmatrix}
R_{32} - R_{23} \\
R_{13} - R_{31} \\
R_{21} - R_{12}
\end{pmatrix}
\end{gather*}
$$

### Gimbal Lock, Singularities, and Quaternion

When using euler angles, certain axes in the body frame could align to each other. E.g., when a plane has a pitch of 90 degrees (as below), its z and x axes are aligned. Then, rotation about z and rotation about x are the same. From this configuration, the plane cannot rotate about the axis that are perpendicular to x,y,z axes, hence it loses 1 degree of freedom.

Mathematically, for a Z-X-Y system,

$$
\begin{gather*}
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos \gamma & -\sin \gamma \\
0 & \sin \gamma & \cos \gamma
\end{bmatrix}
*
\begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
-1 & 0 & 1
\end{bmatrix}
*
\begin{bmatrix}
\cos \alpha & -\sin \alpha & 0 \\
\sin \alpha & \cos \alpha & 0 \\
0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta & -\sin \theta & 0 \\
\sin \theta & \cos \theta & 0 \\
0 & 0 & 1
\end{bmatrix}
\end{gather*}
$$

<p align="center">
<img src="https://github.com/ChengeYang/Probabilistic-Robotics-Algorithms/assets/39393023/5aab7bcb-c434-4ad2-ae41-a4d3314f9dfe" height="200" width="width"/>
</p>

See? The rotation about both the Z axis $\gamma$ and the X axis $\alpha$ will effectively create a combined rotation about the X axis, $\theta$. So, such rotations do not have unique angular values.

### Implementations

- OpenCV: OpenCV provides rotation vector -> single rotation matrix. [See here](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac)

------------------------------------------------------------------------------------------------

## Instantaneous Rotation

According to the Poisson Formula, $R' = Rw^{\land}$, for a small time period $\Delta t$, the ODE can be solved:

$$
\begin{gather*}
R(t) = R(t_0)exp(w^{\land}(t - t_0)) = R(t_0) exp(w^{\land} \Delta t)
\end{gather*}
$$

## Rotation and Skew Matrices in 2D:

In 2D, skew matrix is simply:

$$
\begin{gather*}
\begin{aligned}
& w^{\land} = 
\begin{bmatrix}
0 & -a  \\
a & 0
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

Rotation matrix is:

$$
\begin{gather*}
\begin{aligned}
& R = 
\begin{bmatrix}
cos \theta & -sin \theta \\
sin \theta & cos \theta
\end{bmatrix}
\end{aligned}
\end{gather*}
$$

- One Key property that doesn't hold true in 2D is this: 

$$
\begin{gather*}
\begin{aligned}
& \phi^{\land} R = R (R^T \phi)^{\land}
\end{aligned}
\end{gather*}
$$

But in 2D, one can easily find that: $\phi^{\land} R = R \phi^{\land}$
