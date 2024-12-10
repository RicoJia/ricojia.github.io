---
layout: post
title: Robotics Fundamentals - Rotations
date: '2024-03-10 13:19'
subtitle: Introduction To Rotation Matrices, Lie Group and Lie Algebras, BCH Formula
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

**Representation 3** SO(3)!! This is the most common representation of rotation I've seen so far.

$$
\begin{gather*}
\exp(\hat{\omega}) = I + \frac{sin(\theta)}{\theta} \hat{\omega} +  \frac{1 - cos(\theta)}{\theta} \hat{\omega}^2
\end{gather*}
$$

Common rotation matrices are:

$$
\begin{gather*}
R_x(\gamma) = \begin{bmatrix}
1 & 0 & 0 \\
0 & \cos \gamma & -\sin \gamma \\
0 & \sin \gamma & \cos \gamma
\end{bmatrix}

R_y(\beta) = \begin{bmatrix}
\cos \beta & 0 & \sin \beta \\
0 & 1 & 0 \\
-\sin \beta & 0 & \cos \beta
\end{bmatrix}

R_z(\alpha) = \begin{bmatrix}
\cos \alpha & -\sin \alpha & 0 \\
\sin \alpha & \cos \alpha & 0 \\
0 & 0 & 1
\end{bmatrix}
\end{gather*}
$$

**Representation 4** Quaternion. TODO

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

## Formalize Rotation into A Lie Group

### What Are the Lie Group and Lie Algebras

This is an intuitive introduction of Lie Group and Lie algebras, [inspired by this post](https://physics.stackexchange.com/questions/148116/lie-algebra-in-simple-terms).

A group is a set of algebraic structure (matrices, vectors) that is 1. globally connected to each other, 2. can transition between each other. Formally, a group has:

- Closure: if a and b are in the group, `a op b` is also in the group
- Associativity: `(a op b) op c = a op (b op c)`
- Identity: for any element a, identity element e has `e op a = a`
- Inverses: `a op a_inv = e`

A manifold is a shape that's locally euclidean and globally connected. Integers are not a Lie Group because they are discrete. But real numbers form a consecutive Lie Group.  **A Lie group has to be a smooth manifold, meaning it's differentiable.**
    - Elements in a Lie group can transition to each other through multiplication, inverse, and identity.

Some Lie Group operations are:

A Lie Algebra is a **vector space that captures the infinitesimal structure of a Lie Group, e.g., the tangent space of the Lie Group**. In the context of rotations,

- All 3D rotation matrices form a group called "Special Orthogonal Group, SO(3)", whose definition is $RR^T = I, det(R) = I$
- Their corresponding skew symmetric matrices are their Lie Algebra **so(3)**
  - **Within any Lie algebra, there is Lie Bracket for vectors A and B**: `[A, B] = AB - BA`

### From Skew Symmetric (so(3)) To SO(3)

We can see that R'R^T must be a skew matrix:

$$
\begin{gather*}
RR^T = I => (RR^T)' = R'R^T + RR'^T = 0
\\
R'R^T = -(RR'^T) = -(R'R^T)^T = -\phi(t)^{\land}
\end{gather*}
$$

Where $-\phi(t)^{\land}$ the skew matrix of vector $-\phi(t) = [\phi_1, \phi_2, \phi_1,]$

$$
\begin{gather*}
-\phi(t)^{\land} = \begin{bmatrix}
0 & -\phi_3 & \phi_2 \\
\phi_3 & 0 & -\phi_1 \\
-\phi_2 & \phi_1 & 0
\end{bmatrix}
\end{gather*}
$$

Then, **one can see that taking the direvative of `R` is equivalent to multiplying the skew matrix $-\phi(t)^{\land}$ with it.**
$$
\begin{gather*}
R'R^TR = R' = -\phi(t)^{\land} R
\end{gather*}
$$

Finally, do matrix exponential:

$$
\begin{gather*}
R(t) = exp(-\phi(t)^{\land})
\end{gather*}
$$

### Rodrigues Formula

Let's represent the skew matrix as angle multiplied by a unit vector, $a^{\land}$

$$
\begin{gather*}
-\phi(t) = [\phi_1, \phi_2, \phi_3] = \phi a^{\land}
\end{gather*}
$$

By writing out $a^{\land} a^{\land}$ "Unit" skew matrices have the below two properties:

$$
\begin{gather*}
a^{\land} a^{\land} = aa^T - I
\\
a^{\land} a^{\land} a^{\land} = -a^{\land}
\end{gather*}
$$

- One trick for the second formula is: $a^{\land} (a^{\land} a^{\land}) = a \times (aa^T - I) = -a^{\land}$

So,
$$
\begin{gather*}
exp(\phi^{\land}) = exp(\theta a^{\land}) = \frac{1}{n!}(\theta a^{\land})^n
\\
= I + \theta a^{\land} + \frac{1}{2!} \theta^2 a^{\land} a^{\land} + \frac{1}{3!} \theta^3 a^{\land} a^{\land} a^{\land} + ...
\\
= aa^T - a^{\land} a^{\land} + \theta a^{\land} + \frac{1}{2!} \theta^2 a^{\land} a^{\land} - \frac{1}{3!} \theta^3 a^{\land}...
\\
= aa^T + (\theta - \frac{1}{3!} \theta^3 + ...)a^{\land} - (1 - \frac{1}{2!} \theta^2 + ... )a^{\land} a^{\land}
\\
= aa^T + sin(\theta) a^{\land} -cos(\theta) a^{\land} a^{\land}
\\
= cos(\theta) + (I-cos(\theta)) aa^T + sin(\theta)a^{\land}

\end{gather*}
$$

[Reference](https://jiangren.work/2019/08/09/SLAM%E5%9F%BA%E7%A1%803-%E6%9D%8E%E7%BE%A4%E5%92%8C%E6%9D%8E%E4%BB%A3%E6%95%B0/)

------------------------------------------------------------------------------------------------

## Left and Right Perturbations

Rotations around $t$

Rotation matrices (Lie Algebra) are in the tangent space, not in the Cartesian space. So perturbations in rotation matrices need to be mapped correspondingly.

In general, for the combination of two rotations?

$$
\begin{gather*}
exp^{(C)} = ln(exp(\theta^{\land}) exp^(\phi^{\land})) =
\begin{cases}
J_l^{-1}(\phi)\theta + \phi \text{when perturbation is} \theta \\
J_r^{-1}(\theta)\phi + \theta \text{when perturbation is} \phi \\
\end{cases}
\end{gather*}
$$

TODO: Left perturbation is a "external perturbation" applied on R. For small skew matrix A of rotation axis a, with rotation angle $\theta$ and unit vector `n`

$$
\begin{gather*}
J_l = \frac{(\partial{exp^{(A)}} R)}{A}
\\
TODO??
\\
J_l = \frac{sin(\theta)}{\theta}I + (1-\frac{sin(\theta)}{\theta})nn^T + \frac{1-cos(\theta)}{\theta}A
\\ J_r = J_l(-\theta)
\\TODO
\end{gather*}
$$

TODO: right perturbation is an "internal perturbation" that's applied on the points / vector R has been applied to.

## Useful Formulas

Here, assume we have rotation $\phi$, and its corresponding rotation, $R$,

$$
\begin{gather*}
exp(\phi) = R   \\
log(R) = \phi
\end{gather*}
$$

### BCH (Baker-Cambell-Hausdorff) Formula

BCH formula states that for two Lie Algebras $e^{A}$ and $e^{B}$ with skew matrices `A` and `B` that composes `C`: $C = log(e^{A} e^{B})$, Then to determine C

$$
\begin{gather*}
C = A + B + \frac{1}{2}[A, B] + \frac{1}{12}([A, [A, B] + [B, [B, A]]])
\end{gather*}
$$

Where `[A, B]` is a commutator that $[A, B] = AB - BA$, and $[A, [A, B]] = A(AB-BA) - (AB-BA)A$

#### Applications of BCH

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
