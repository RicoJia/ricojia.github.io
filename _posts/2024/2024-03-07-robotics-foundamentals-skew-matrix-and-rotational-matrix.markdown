---
layout: post
title: Robotics Fundamentals - Skew Matrix and Rotational Matrix
date: '2024-03-10 13:19'
subtitle: Introduction To Rotation Matrices, Lie Group and Lie Algebras
comments: true
tags:
    - Robotics
---

## SO(3)

SO(3) is the most common representation of rotation I've seen so far.

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

## Skew Symmetric Matrices

Let $\phi \in R^3$ be a vector, and let $R \in SO(3)$ be a rotation matrix (i.e., $R^TR = I$ and $det⁡(R)=1$). Define the “hat” (or wedge) operator $\phi^{\land}$ as the skew-symmetric matrix satisfying

$$
\begin{gather*}
\begin{aligned}
& \phi^{\land} x = \phi \times x
\end{aligned}
\end{gather*}
$$

for any $x \in R^3$

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

#### Lie Definitions

A Lie Algebra is a **vector space that captures the infinitesimal structure of a Lie Group, e.g., the tangent space of the Lie Group**. In the context of rotations,

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/a66f9f2e-afbb-4f87-bd15-052e3633da60" height="300" alt=""/>
            <figcaption><a href="https://johnwlambert.github.io/lie-groups/">Source: John Lambert</a></figcaption>
       </figure>
    </p>
</div>

- All 3D rotation matrices form a group called "Special Orthogonal Group, SO(3)", whose definition is $RR^T = I, det(R) = I$
- **Their corresponding skew symmetric matrices are their Lie Algebra so(3)**. The exponential $exp(\phi^{\land}) = R$ is called "exponential mapping"
  - **Within any Lie algebra, there is Lie Bracket for vectors A and B**: `[A, B] = AB - BA`

### From Skew Symmetric `so(3)` To SO(3)

We can see that R'R^T must be a skew matrix:

$$
\begin{gather*}
\begin{aligned}
& RR^T = I
\\ & \Rightarrow (RR^T)' = R'R^T + RR'^T = 0
\\ & \Rightarrow R'R^T = -(RR'^T) = -(R'R^T)^T = \phi(t)^{\land}
\end{aligned}
\end{gather*}
$$

Where $\phi(t)^{\land}$ the skew matrix of vector $\phi(t) = [\phi_1, \phi_2, \phi_1,]$

$$
\begin{gather*}
\phi(t)^{\land} = \begin{bmatrix}
0 & -\phi_3 & \phi_2 \\
\phi_3 & 0 & -\phi_1 \\
-\phi_2 & \phi_1 & 0
\end{bmatrix}
\end{gather*}
$$

Then, **one can see that taking the direvative of `R` is equivalent to multiplying the skew matrix $\phi(t)^{\land}$ with it.**
$$
\begin{gather*}
R'R^TR = R' = \phi(t)^{\land} R
\end{gather*}
$$

The above is also known as the Poisson Formula. If we define

$$
\begin{gather*}
w^{\land} = R^{T}R'
\end{gather*}
$$

We have another form of the **Poisson Formula**:

$$
\begin{gather*}
R' = Rw^{\land}
\end{gather*}
$$

Finally, do matrix exponential:

$$
\begin{gather*}
R(t) = exp(\phi(t)^{\land})
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
\begin{aligned}
& exp(\phi^{\land}) = exp(\theta a^{\land}) = \frac{1}{n!}(\theta a^{\land})^n
\\
& = I + \theta a^{\land} + \frac{1}{2!} \theta^2 a^{\land} a^{\land} + \frac{1}{3!} \theta^3 a^{\land} a^{\land} a^{\land} + ...
\\
& = aa^T - a^{\land} a^{\land} + \theta a^{\land} + \frac{1}{2!} \theta^2 a^{\land} a^{\land} - \frac{1}{3!} \theta^3 a^{\land}...
\\
& = aa^T + (\theta - \frac{1}{3!} \theta^3 + ...)a^{\land} - (1 - \frac{1}{2!} \theta^2 + ... )a^{\land} a^{\land}
\\
& = aa^T + sin(\theta) a^{\land} -cos(\theta) a^{\land} a^{\land}
\\
& = cos(\theta) + (I-cos(\theta)) aa^T + sin(\theta)a^{\land}
\\
Or
\\
& = I + (1-cos \theta) a^{\land} a^{\land} + sin \theta a^{\land}
\end{aligned}
\end{gather*}
$$

[Reference](https://jiangren.work/2019/08/09/SLAM%E5%9F%BA%E7%A1%803-%E6%9D%8E%E7%BE%A4%E5%92%8C%E6%9D%8E%E4%BB%A3%E6%95%B0/)


## Properties

- $ \phi^{\land} R = R (R^T \phi)^{\land}$

$$
\begin{gather*}
\begin{aligned}
&
\end{aligned}
\end{gather*}
$$

