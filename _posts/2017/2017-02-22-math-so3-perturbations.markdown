---
layout: post
title: Math - SO(3) Perturbations
date: '2017-02-22 13:19'
subtitle: Left Perturbation, Right Perturbation, BCH Formula
comments: true
tags:
    - Math
---

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

$R_{wc1}R_{c1c2}$ is called a "right perturbation". This is **more common** since we are always more interested in world frame coordinates.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/081626cc-89db-43cf-a4fa-3cf353a7bc25" height="300" alt=""/>
        <figcaption><a href="https://alida.tistory.com/73">Source</a></figcaption>
    </figure>
</p>
</div>


### Perturbations Can Not Be Added On `so(3)` Or `SO(3)`

One question is can skew matrices `so(3)` be added?

$$
\begin{gather*}
R = exp(\phi_1^{\land}) exp(\phi_2^{\land}) = exp(\phi_1^{\land} + \phi_2^{\land})?
\end{gather*}
$$

In 2D, the above form is correct. However, in 3D, we know that $R_1R_2 \ne R_2R_1$, so the above can't be true. Rotation matrices (Lie Algebra) are in the tangent space, not in the Cartesian space. So perturbations in rotation matrices need to be mapped correspondingly.

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
\begin{aligned}
& A = \begin{bmatrix}
0 & -a  \\
a & 0
\end{bmatrix}
\\ &
B = \begin{bmatrix}
0 & -b  \\
b & 0
\end{bmatrix}

\\ &
[A, B] = AB - BA = 0

\\ &
C = A + B + \frac{1}{2}[A, B] = \begin{bmatrix}
0 & -(a+b)  \\
a+b & 0
\end{bmatrix}

\end{aligned}
\end{gather*}
$$

#### How BCH Approximates The Left Perturbation

I'm omitting the derivation of below `SO(3)` BCH linerization formulae because it's not trivial.

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
\begin{aligned}
&
A = \theta a
\\ &
R = exp(A^{\land})
\\ &
= I + (1-cos \theta) a^{\land} a^{\land} + sin \theta a^{\land}
\\ &
= I + \frac{(1-cos \theta) A^{\land} A^{\land}}{\theta^2} + \frac{sin \theta A^{\land}}{\theta}  
\end{aligned}
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
\begin{aligned}
& exp^{(C)} = ln(exp(A^{\land}) exp((\Delta A)^{\land}))
\\ &
= J_r^{-1}(A) \Delta A + A
\\ &
J_r(-\theta) = J_l(\theta) 
\end{aligned}
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




