---
layout: post
title: Robotics - [IMU Pre-integration Model 3] Noise Model
date: '2024-04-01 13:19'
subtitle: Jacobian Derivation
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Pre-integration Model

### Rotation Model

Using the BCH approximation:

$$
\begin{gather*}
\begin{aligned}
& exp((\Delta B + B)^{\land}) \approx exp((J_l^{-1}(B) \Delta B) ^{\land}) exp(B^{\land})
\end{aligned}
\end{gather*}
$$

- We can separate the noise terms from the measurement terms of the rotation:

$$
\begin{gather*}
\begin{aligned}
& \Delta R_{ij} := R_i^T R_j = \prod_{k=i}^{j-1} (Exp((\tilde{w_k} - b_{g,k} - \eta_{gd, k})\Delta t))
\\ &
\approx R_i^T R_j  \prod_{k=i}^{j-1} (Exp((\tilde{w_k} - b_{g,k})\Delta t)  Exp( -J_l^{-1} \eta_{gd, k} )\Delta t)
\end{aligned}
\end{gather*}
$$

- Measured rotation part is: $\Delta  \tilde{R_{ij}} = \prod_{k=i}^{j-1} Exp((\tilde{w_k} - b_{g,k}) \Delta t)$.

- Using:

$$
\begin{gather*}
\begin{aligned}
& Exp(-J_{r,i}\eta_{gd,i}\Delta t)\Delta \tilde{R}_{i+1, i+2} = \Delta \tilde{R}_{i+1, i+2}Exp(- \Delta \tilde{R}_{i+1, i+2} ^T J_{r,i}\eta_{gd,i}\Delta t)
\end{aligned}
\end{gather*}
$$

The above becomes:

$$
\begin{gather*}
\begin{aligned}
& \Delta R_{ij} = Exp \left( (\tilde{\omega}_i - b_{g,i}) \Delta t \right)
Exp \left( -J_{r,i} \eta_{gd,i} \Delta t \right)
\underbrace{Exp \left( (\tilde{\omega}_{i+1} - b_{g,i}) \Delta t \right)}_{\Delta \tilde{R}_{i+1,i+2}}
Exp \left( -J_{r,i+1} \eta_{gd,i} \Delta t \right)\cdots ,

\\ &
= \Delta \tilde{R}_{i,i+1}
Exp \left( -J_{r,i} \eta_{gd,i} \Delta t \right)
\Delta \tilde{R}_{i+1,i+2}
Exp \left( -J_{r,i+1} \eta_{gd,i} \Delta t \right) \cdots ,

\\ &
= \Delta \tilde{R}_{i,i+2}
Exp \left( -\Delta \tilde{R}_{i+1,i+2}^\top J_{r,i} \eta_{gd,i} \Delta t \right)
Exp \left( -J_{r,i+1} \eta_{gd,i} \Delta t \right) \cdots ,

\\ &
= \Delta \tilde{R}_{i,i+2}
Exp \left( -\Delta \tilde{R}_{i+1,i+2}^\top J_{r,i} \eta_{gd,i} \Delta t \right)
\Delta \tilde{R}_{i+2,i+3} \cdots .

\\ &
= \Delta \tilde{R}_{i,j} \prod_{k=i}^{j-1}
Exp \left( -\Delta \tilde{R}_{k,k+1}^\top J_{r,i} \eta_{gd,i} \Delta t \right) \cdots

\\ &
= \Delta \tilde{R}_{i,j} Exp \left(-\delta \phi_{i,j} \right)
\end{aligned}
\end{gather*}
$$

Where the accumulated observed rotation part is $\Delta \tilde{R}_{i,j}$

#### The above is to use [this property](https://ricojia.github.io/2017/02/22/lie-group/#3-rt-textexpphi-r--textexprt-phi) to move rotation matrices to the right

$$
\begin{gather*}
\begin{aligned}
& R^T \text{Exp}(\phi) R = \text{Exp}(R^T \phi)
\end{aligned}
\end{gather*}
$$

### Velocity Model

For velocity, we plug the above into the formula. Similarly, we apply the first order taylor approximation of $Exp(-\delta \phi) \approx (I - \delta \phi)$, and drop second order small terms:

$$
\begin{gather*}
\begin{aligned}
& \Delta v_{ij} := R_i^T(v_j - v_i - g_k \Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta R_{ik} (\tilde{a_k} - b_{a,k} - \eta_{ad, k}) \Delta t

\\ &
= \sum_{k=i}^{j-1} \Delta \tilde{R}_{i,k} Exp \left(-\delta \phi_{i,k} \right) (\tilde{a_k} - b_{a,k} - \eta_{ad, k}) \Delta t

\\ &
\approx \sum_{k=i}^{j-1} \Delta \tilde{R}_{i,k} (I -\delta \phi_{i,k}) (\tilde{a_k} - b_{a,k} - \eta_{ad, k}) \Delta t

\\&
= \sum_{k=i}^{j-1} \Delta \tilde{R}_{i,k}(\tilde{a_k} - b_{a,k})  \Delta t + \Delta \tilde{R}_{i,k} (\tilde{a_k} - b_{a,k} - \eta_{ad, k})^{\land} \phi_{i,k} \Delta t - \Delta \tilde{R}_{i,k} \eta_{ad, k}

\\ &
\text{Defining velocity observation:}

\\ &
\Delta \tilde{v_{ij}} = \sum_{k=i}^{j-1} \Delta \tilde{R}_{i,k}(\tilde{a_k} - b_{a,k})  \Delta t

\\ &
\text{Omitting second order term} - \eta_{ad, k}^{\land} \phi_{i,k},

\\ &
\rightarrow = \Delta \tilde{v_{ij}} +  \sum_{k=i}^{j-1} \Delta \tilde{R}_{i,k} (\tilde{a_k} - b_{a,k} )^{\land} \phi_{i,k} \Delta t - \Delta \tilde{R}_{i,k} \eta_{ad, k} \Delta t

\\ &
=  \Delta \tilde{v_{ij}} - \delta v_{i,j}

\end{aligned}
\end{gather*}
$$

### Position Model

For position, we plug the above into the formula. Similarly, we apply the first order taylor approximation of $Exp(-\delta \phi) \approx (I - \delta \phi)$, and drop second order small terms:

$$
\begin{gather*}
\begin{aligned}
&
\Delta p_{ij} := R_i^T(p_j - p_i - v_i \Delta t_{ij} - \frac{1}{2} g_k \Delta t_{ij}^2) =

\\ &

= \sum_{k=i}^{j-1} \Delta v_{ik} \Delta t + \frac{1}{2} \Delta R_{ik} (\tilde{a_k} - b_{a,k} - \eta_{ad, k}) \Delta t^2

\\ &
= \sum_{k=i}^{j-1} (\Delta \tilde{v_{ij}} - \delta v_{i,j} )\Delta t + \frac{1}{2} \Delta \tilde{R}_{i,k} (\tilde{a_k} - b_{a,k})\Delta t^2

\\ &
- \delta v_{ik} \Delta t + \frac{1}{2} \Delta \tilde{R}_{i,k} (\tilde{a_k} - b_{a,k})^{\land}\delta \phi \Delta t^2 - \frac{1}{2} \Delta \tilde{R}_{i,k} \eta_{ad, k} \Delta t^2

\\ &
\text{Define accumulated position part observation:}

\\ &
\Delta \tilde{p_{i,j}} = \sum_{k=i}^{j-1}[\Delta v_{i,k} \Delta t] + \frac{1}{2} \Delta \tilde{R}_{i,k} (\tilde{a_k} - b_{a,k})\Delta t^2

\\ &
\text{The above becomes:}

\\ &
= \Delta \tilde{p_{i,j}} + \sum_{k=i}^{j-1} - \delta v_{ik} \Delta t + \frac{1}{2} \Delta \tilde{R}_{i,k} (\tilde{a_k} - b_{a,k})^{\land}\delta \phi \Delta t^2 - \frac{1}{2} \Delta \tilde{R}_{i,k} \eta_{ad, k} \Delta t^2

\\ &
:= \Delta \tilde{p_{i,j}} - \delta p_{i,j}
\end{aligned}
\end{gather*}
$$

To further analyze their noises, the right-hand-side of the accumlated observations can also be written in terms of their true values and the noises

$$
\begin{gather*}
\begin{aligned}
& \Delta \tilde{R_{ij}} := R_i^T R_j Exp(\delta \phi_{ij})

\\ &

\Delta \tilde{v_{ij}} =  R_i^T(v_j - v_i - g_k \Delta t_{ij}) + \delta v_{i,j}

\\ &
\Delta \tilde{p_{i,j}} = R_i^T(p_j - p_i - v_i \Delta t_{ij} - \frac{1}{2} g_k \Delta t_{ij}^2) + \delta p_{i,j}
\end{aligned}
\end{gather*}
$$

So, in short, the accumulated observations can be reasonably easy to add up / multiply. The right-hand-side of the accumlated observations are easy to form edges in a graph for Least-Square-Error problems between nodes. Now the question is, are the noises Gaussian? If so, how large are they?

## IMU Preintegration Noise Model

### Accumulated Rotation Noise

$$
\begin{gather*}
\begin{aligned}
& Exp \left(-\delta \phi_{i,j} \right) = \prod_{k=i}^{j-1}
Exp \left( -\Delta \tilde{R}_{k,k+1}^\top J_{r,i} \eta_{gd,i} \Delta t \right)

\end{aligned}
\end{gather*}
$$

Now let's get

$$
\begin{gather*}
\begin{aligned}
& \phi_{ij} = -Log(\prod_{k=i}^{j-1}
Exp \left( -\Delta \tilde{R}_{k,k+1}^\top J_{r,i} \eta_{gd,i} \Delta t \right))
\end{aligned}
\end{gather*}
$$

By using BCH for right perturbation:

$$
\begin{gather*}
C = ln(exp(A^{\land}) exp(B^{\land})) =
J_r^{-1}(A)B + A
\end{gather*}
$$

And that each noise angle themselves are small, we know the Jacobians are almost identity. So we get:

$$
\begin{gather*}
\begin{aligned}
& \phi_{ij} \approx \sum_{k=i}^{j} \Delta \tilde{R}_{k,k+1}^\top J_{r,i} \eta_{gd,i} \Delta t
\end{aligned}
\end{gather*}
$$

The mean is only a linear combination of with zero-mean gaussian noise $\eta_{gd,i}$, so the mean is zero. Now let's get covariance. We can show that the covariance is a recursive form, too.

$$
\begin{gather*}
\begin{aligned}
& \phi_{ij} \approx \sum_{k=i}^{j} \Delta \tilde{R}_{k,k+1}^\top J_{r,i} \eta_{gd,i} \Delta t
\\ &
= \sum_{k=i}^{j-2} \tilde{\Delta R}_{k+1,j}^{\top} J_{r,k} \eta_{gd,k} \Delta t + \underbrace{\Delta R_{j,j}^{\top}}_{=I} J_{r,j-1} \eta_{gd,j-1} \Delta t,

\\ &

= \sum_{k=i}^{j-2} \tilde{\Delta R}_{k+1,j}^{\top} J_{r,k} \eta_{gd,k} \Delta t + J_{r,j-1} \eta_{gd,j-1} \Delta t,

\\ & \text{Since:}
\tilde{\Delta R}_{k+1,j}^{\top} = \left( \tilde{\Delta R}_{k+1,j-1} \tilde{\Delta R}_{j-1,j} \right)^{\top}

\\ &
= \tilde{\Delta R}_{j-1,j}^{\top} \sum_{k=i}^{j-2} \tilde{\Delta R}_{k+1,j}^{\top} J_{r,k} \eta_{gd,k} \Delta t + J_{r,j-1} \eta_{gd,j-1} \Delta t,

\\ &
= \tilde{\Delta R}_{j-1,j}^{\top} \delta \phi_{i,j-1} + J_{r,j-1} \eta_{gd,j-1} \Delta t.
\end{aligned}
\end{gather*}
$$

This is a linear system. Using the covariance of mulplied matrix: $cov(AX) = A cov(X) A^T$, we can see that **the covariance keeps growing** if we accumulate:

$$
\begin{gather*}
\begin{aligned}
& \Sigma_j =  \Delta \tilde{R}_{j-1, j}^T \Sigma_{j-1} \Delta \tilde{R}_{j-1, j} + J_{r, j-1} \Sigma_{\eta_{gd}} J_{r, j-1}^T \Delta t^2
\end{aligned}
\end{gather*}
$$

And this covariance growth makes sense.

### Accumulated Velocity Noise

If we find the recursive form of the noise:

$$
\begin{gather*}
\begin{aligned}
& \delta v_{ij} = \sum_{k=i}^{j-1} - \Delta \tilde{R}_{i,k} (\tilde{a_k} - b_{a,k})^{\land} \phi_{i,k} \Delta t + \Delta \tilde{R}_{i,k} \eta_{ad, k} \Delta t

\\ &
= \sum_{k=i}^{j-2} \left[ -\tilde{\Delta R}_{ik} (\tilde{a}_k - b_{a,i})^\wedge \delta \phi_{ik} \Delta t + \tilde{\Delta R}_{ik} \eta_{ad,k} \Delta t \right]

\\ &

\quad - \tilde{\Delta R}_{i,j-1} (\tilde{a}_{j-1} - b_{a,i})^\wedge \delta \phi_{i,j-1} \Delta t + \tilde{\Delta R}_{i,j-1} \eta_{ad,j-1} \Delta t,

\\ &
= \delta v_{i,j-1} - \tilde{\Delta R}_{i,j-1} (\tilde{a}_{j-1} - b_{a,i})^\wedge \delta \phi_{i,j-1} \Delta t + \tilde{\Delta R}_{i,j-1} \eta_{ad,j-1} \Delta t.
\end{aligned}
\end{gather*}
$$

This may or may not increase.

### Accumulated Position Noise

$$
\begin{gather*}
\begin{aligned}
& \delta p_{i,j} =  \sum_{k=i}^{j-1}  \delta v_{ik} \Delta t - \frac{1}{2} \Delta \tilde{R}_{i,k} (\tilde{a_k} - b_{a,k})^{\land}\delta \phi \Delta t^2 + \frac{1}{2} \Delta \tilde{R}_{i,k} \eta_{ad, k} \Delta t^2

\\ &
= \sum_{k=i}^{j-2} \left[ \delta v_{ik} \Delta t - \frac{1}{2} \tilde{\Delta R}_{ik} (\tilde{a}_k - b_{a,i})^\wedge \delta \phi_{ik} \Delta t^2 + \frac{1}{2} \tilde{\Delta R}_{ik} \eta_{ad,k} \Delta t^2 \right]

\\ &
\quad + \delta v_{i,j-1} \Delta t - \frac{1}{2} \tilde{\Delta R}_{i,j-1} (\tilde{a}_{j-1} - b_{a,i})^\wedge \delta \phi_{i,j-1} \Delta t^2 + \frac{1}{2} \tilde{\Delta R}_{i,j-1} \eta_{ad,j-1} \Delta t^2,

\\ &
= \delta p_{i,j-1} + \delta v_{i,j-1} \Delta t - \frac{1}{2} \tilde{\Delta R}_{i,j-1} (\tilde{a}_{j-1} - b_{a,i})^\wedge \delta \phi_{i,j-1} \Delta t^2 + \frac{1}{2} \tilde{\Delta R}_{i,j-1} \eta_{ad,j-1} \Delta t^2.
\end{aligned}
\end{gather*}
$$

### Accumulated Noise Model All Together

If we put the accumulated noises into a vector $\eta_{ik}$

$$
\begin{gather*}
\begin{aligned}
& \eta_{ik} =
\begin{bmatrix}
\delta \phi_{ik} \\
\delta v_{ik} \\
\delta p_{ik}
\end{bmatrix},
\end{aligned}
\end{gather*}
$$

Noise of biases into a vector:

$$
\begin{gather*}
\begin{aligned}
& \eta_{d,j} =
\begin{bmatrix}
\eta_{gd,j} \\
\eta_{ad,j}
\end{bmatrix},
\end{aligned}
\end{gather*}
$$

The recursive form of the accumulated noises are:

$$
\begin{gather*}
\begin{aligned}
& \eta_{ij} = \mathbf{A}_{j-1} \eta_{i,j-1} + \mathbf{B}_{j-1} \eta_{d,j-1},
\end{aligned}
\end{gather*}
$$

Where:

$$
\begin{gather*}
\begin{aligned}
& A_{j-1} =
\begin{bmatrix}
\tilde{\Delta R}_{j-1,j}^{\top} & 0 & 0 \\
-\tilde{\Delta R}_{i,j-1} (\tilde{a}_{j-1} - b_{a,i})^\wedge \Delta t & I & 0 \\
-\frac{1}{2} \tilde{\Delta R}_{i,j-1} (\tilde{a}_{j-1} - b_{a,i})^\wedge \Delta t^2 & \Delta t I & I
\end{bmatrix},

B_{j-1} =
\begin{bmatrix}
J_{r,j-1} \Delta t & 0 \\
0 & \tilde{\Delta R}_{i,j-1} \Delta t \\
0 & \frac{1}{2} \tilde{\Delta R}_{i,j-1} \Delta t^2
\end{bmatrix}.
\end{aligned}
\end{gather*}
$$

The covariance is accumulated as well:

$$
\begin{gather*}
\begin{aligned}
& \Sigma_{i,k+1} = A_{k+1} \Sigma_{i,k} A_{k+1}^{\top} + B_{k+1} \text{Cov}(\eta_{d,k}) B_{k+1}^{\top},
\end{aligned}
\end{gather*}
$$

Note that $A_{k+1}$ is close to identity, rotational noises are solely added up by incremental rotational noises. Noises of the velocity and positional parts primarily come from themselves.
