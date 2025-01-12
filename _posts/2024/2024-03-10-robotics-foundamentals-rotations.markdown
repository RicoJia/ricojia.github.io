---
layout: post
title: Robotics Fundamentals - Rotations
date: '2024-03-10 13:19'
subtitle: BCH Formula, Instantaneous Rotation
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

**Representation 3** [SO(3)](./2024-03-07-robotics-foundamentals-skew-matrix-and-rotational-matrix.markdown)


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

Mathematically,

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

See? If in the plane's control system orientation is represented by Z-X-Y euler angles, when the plane is in this initial position, rotating about X will affect the reading of yaw (We would want the to be independent at all times).

Quaternion

### Implementations

- OpenCV: OpenCV provides rotation vector -> single rotation matrix. [See here](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac)

------------------------------------------------------------------------------------------------

## Left and Right Perturbations

Imagine we have a world frame, a car frame, and a pedestrian. Now the car has rotated from the `car1` pose to the `car2` pose. We assume the pedestrain has not moved:

In `car1` and `car2` poses, the pedestrian is at
$$
\begin{gather*}
p_1 = R_{c1w}p_w
\\
p_2 = R_{c2c1} R_{c1w}p_w
\end{gather*}
$$

$R_{c2c1} R_{c1w}$ is called a "left perturbation".

On the other hand, at the world frame,

$$
\begin{gather*}
p_w = R_{wc1}p_1
\\
p_w = R_{wc1}R_{c1c2}p_2
\end{gather*}
$$

$R_{wc1}R_{c1c2}$ is called a "right perturbation". This is **the more common** since we are always more interested in world frame coordinates.

### Perturbations Can Not Be Added On `so(3)` Or `SO(3)`

One question is can skew matrices `so(3)` be added?

$$
\begin{gather*}
R = exp(\phi_1^{\land}) exp(\phi_2^{\land}) = exp(\phi_1^{\land} + \phi_2^{\land})?
\end{gather*}
$$

However, the BCH formula tells us that it's not the case. Rotation matrices (Lie Algebra) are in the tangent space, not in the Cartesian space. So perturbations in rotation matrices need to be mapped correspondingly.

### BCH (Baker-Cambell-Hausdorff) Formula

BCH formula states that for two Lie Algebras $e^{A}$ and $e^{B}$ with skew matrices `A` and `B` that composes `C`: $R = e^{C} = e^{A^{\land}} e^{B^{\land}}$, Then to determine C

$$
\begin{gather*}
C = A + B + \frac{1}{2}[A, B] + \frac{1}{12}([A, [A, B] - [B, [B, A]]]) ...
\end{gather*}
$$

Where `[A, B]` is a commutator that $[A, B] = AB - BA$, and $[A, [A, B]] = A(AB-BA) - (AB-BA)A$. If we take the taylor expansion with perturbations as shown, we get:

#### Small Example of BCH

- For small $A$, $B$, one have $C \approx A + B + \frac{1}{2}[A, B]$. Example: for two rotations `A` and `B`

$$
\begin{gather*}
A = \begin{bmatrix}
0 & -a  \\
a & 0
\end{bmatrix}
\\
B = \begin{bmatrix}
0 & -b  \\
b & 0
\end{bmatrix}

\\
[A, B] = AB - BA = 0
\\
C = A + B + \frac{1}{2}[A, B] = \begin{bmatrix}
0 & -(a+b)  \\
a+b & 0
\end{bmatrix}

\end{gather*}
$$

#### How BCH Approximates The Left Perturbation

$$
\begin{gather*}
exp^{(C)} = ln(exp(A^{\land}) exp^(B^{\land})) =
\begin{cases}
J_l^{-1}(B)A + B \text{when perturbation (small value) is} A \\
J_r^{-1}(A)B + A \text{when perturbation (small value) is} B \\
\end{cases}
\end{gather*}
$$

This approximation linearizes manipulation to skew matrices addition. To see how this is derived:

- Rodrigues Formula can also be written as: (**A is a rotation vector**)

$$
\begin{gather*}
\\
A = \theta a
\\
R = exp(A^{\land})
\\
= I + (1-cos \theta) a^{\land} a^{\land} + sin \theta a^{\land}
\\
= I + \frac{(1-cos \theta) A^{\land} A^{\land}}{\theta^2} + \frac{sin \theta A^{\land}}{\theta}  
\end{gather*}
$$

Left Jacobian is defined as the "derivative" that measures the infinitesimally small change in R w.r.t to $A^{\land}$. So it's

$$
\begin{gather*}
J_{l}(A) =  \frac{\partial{exp(A^{\land})}}{\partial{A}}
\end{gather*}
$$

So note that in $exp(A^{\land}) = I + \frac{(1-cos \theta) A^{\land} A^{\land}}{\theta^2} + \frac{sin \theta A^{\land}}{\theta}$, $\frac{\partial A^{\land}}{A}$ is not hard because TODO?

$\frac{\theta}{A}$ is a bit tricky. But we have $\theta = \sqrt{A_1^2 + A_2^2 + A_3^2}$. So

$$
\begin{gather*}
\frac{\partial{\theta}}{\partial{A}} = \frac{A}{\theta} = a
\end{gather*}
$$

Eventually, we get

$$
\begin{gather*}
J_{l}(A) = \frac{\partial{e^{A^{\land}}}}{\partial{A}} = \frac{sin \theta}{\theta} I + (1 - \frac{sin \theta}{\theta}) aa^T + \frac{1 - cos\theta}{\theta} a^{\land}

\end{gather*}
$$

#### Right Perturbation

$$
\begin{gather*}
=> J_{l}(\theta)^{-1} = \frac{\theta}{2}cot \frac{\theta}{2} I + (1 - \frac{\theta}{2}cot \frac{\theta}{2}) aa^T - \frac{\theta}{2} a^{\land}
\end{gather*}
$$

And the right Jacobian is:
$$
\begin{gather*}
exp^{(C)} = ln(exp(A^{\land}) exp((\Delta A)^{\land}))
\\
= J_r^{-1}(A) \Delta A + A
\\
J_r(-\theta) = J_l(\theta)   // TODO?
\end{gather*}
$$

#### Perturbations All Together

$$
\begin{gather*}
exp(C) = exp(\Delta B^{\land}) exp(B^{\land}) = exp((B + J_l^{-1}(\Delta B) B)^{\land})
\end{gather*}
$$

So when we want to add a small value vector together:

$$
\begin{gather*}
exp((\Delta B + B)^{\land}) = exp((J_l^{-1}(B) \Delta B) ^{\land}) exp(B^{\land})
\end{gather*}
$$

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
