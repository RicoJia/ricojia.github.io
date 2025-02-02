---
layout: post
title: Robotics - IMU Pre-integration Model
date: '2024-03-24 13:19'
subtitle: Pre-Integration Model
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Motivation

In [previous posts](./2024-03-25-robotics-full-eskf.markdown), we have seen the ESKF framework, where IMU data is fused with observations (GNSS, encoders) following this incremenetal order:

```
ESKF Prediction with IMU | ESKF Prediction with IMU | ESKF update with GPS ....
```

ESKF's prediction is done one by one on each IMU input data. However, imagine we have a graph optimization process that optimizes all IMU and sparse GPS data within a time frame. **Everytime we adjust state variables, we need to recalculate all ESKF state variables on every iteration, which is very inefficient:**

1. Start at time $t_0$ with some initial guess $x^0$
2. Integrate IMU step-by-step up to time $t_N$
3. Apply one iteration of the GPS constraints, correct the states.
4. Re-run the same step-by-step ESKF from t0t0​ to tNtN​ with updated states, do another iteration, and so on…

More specifically, in 2, the true values we try to model are:

$$
\begin{gather*}
& R_j = R_i \prod_{k=i}^{j-1} (Exp((\tilde{w_k} - b_{g,k} - \eta_{gd, k})\Delta t))
\\ &
v_j = v_i +g_k \Delta t_{ij} + \sum_{k=i}^{j-1} R_k (\tilde{a_k} - b_{a,k} - \eta_{ad, k}) \Delta t
\\ &
p_j = p_i + \sum_{k=i}^{j-1} v_k \Delta t + \frac{1}{2} g_k \Delta t_{ij}^2 + \frac{1}{2} \sum_{k=i}^{j-1} R_k (\tilde{a_k} - b_{a,k} - \eta_{ad, k}) \Delta t^2 
\tag{1}
\end{gather*}
$$

- Rotation is following the right perturbation model $R_1 = R_0 \Delta R$ because angular velocity $w$ is observed in frame `0`. In other words, $\Delta R = R_{0,1}$
- This is "direct integration". One can see that the integration terms have $R_j$, which is a state at a specific time. This is why we "need to recalculate" state variables

Pre-integration is a standard method that's used in tightly-coupled LIO and VIO systems. We can easily find intermidate state variables (pre-integration factors) that do not rely on state variables? (TODO, and review below as well)

1. Pre-integrate all measurements into compact pre-integration factors.
2. On each optimizer iteration, upon receving an update, apply the linearized updates to state variables. 

## Definitions of Intermediate Variables

To make $(1)$ easier, we accumulate intermediate values that are separate from absolute state values. These intermediate values are defined by moving absolute state terms to the left. These are what pre-integration integrates. Note that they share the same physical units as the rotation, velocity, and position, but they are just intermediate values without clear physical meanings: 

$$
\begin{gather*}
\begin{aligned}
& \Delta R_{ij} := R_i^T R_j = \prod_{k=i}^{j-1} (Exp((\tilde{w_k} - b_{g,k} - \eta_{gd, k})\Delta t))
\\ &
\Delta v_{ij} := R_i^T(v_j - v_i - g_k \Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta R_{ik} (\tilde{a_k} - b_{a,k} - \eta_{ad, k}) \Delta t
\\ &
\Delta p_{ij} := R_i^T(p_j - p_i - v_i \Delta t_{ij} - \frac{1}{2} g_k \Delta t_{ij}^2) = \sum_{k=i}^{j-1} \Delta v_{ik} \Delta t + \frac{1}{2} \sum_{k=i}^{j-1} \Delta R_{ik} (\tilde{a_k} - b_{a,k} - \eta_{ad, k}) \Delta t^2 
\end{aligned}
\end{gather*}
$$

- The "rotation part" $\Delta R_{ij}$ is the accumulated rotation between i, j
- The "velocity part" $\Delta v_{ij}$ and the "position part" $\Delta p_{ij}$ are less intuitive. But all three values start at 0 at ith time. They are in the forms of product or sum, which makes later linearization with Jacobian easier.
    - With linearization, we can just apply correction terms based if bias terms like $b_{a} changes.
- All three values are independent of absolute state variables

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
& \Delta R_{ij} := R_i^T R_j \prod_{k=i}^{j-1} (Exp((\tilde{w_k} - b_{g,k} - \eta_{gd, k})\Delta t))
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

#### The above is to use [this property](../2017/2017-02-19-lie-group.markdown) to move rotation matrices to the right:

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

## Update Pre-integration When Updating Biases

Pre-integration parts are functions are functions `w.r.t` gyro and acceleration biases: $b_{g,i}, b_{a,i}$. In graph optimization, we would usually need to update these bias terms. So how do we update the preintegration terms? **The trick is again, linearization: we assume each pre-integration term can be approximated linearly**

### Jacobian of Rotational Part w.r.t Gyro Bias 

Recall:

$$
\begin{gather*}
\begin{aligned}
& \tilde{\Delta R_{ij}} = \prod_{k=i}^{j-1} Exp((\tilde{w_k} - b_{g,k}) \Delta t)

\\ &
\rightarrow \text{with bias update}

\\ &
\tilde{\Delta R_{ij}}(b_{g,i} + \delta b_{g,i}) = \prod_{k=i}^{j-1} Exp((\tilde{w_k} - b_{g,k}  - \delta b_{g,i}) \Delta t) :\approx \tilde{\Delta R_{ij}}(b_{g,i}) Exp(\frac{\partial \Delta R_{ij}}{\partial b_{g,i}} \delta b_{g,i})

\rightarrow

\\ &
\Delta \tilde{R}_{i,j}(b_{g,i} + \delta b_{g,i}) =
\prod_{k=i}^{j-1} \text{Exp} \left( (\tilde{\omega}_k - (b_{g,i} + \delta b_{g,i})) \Delta t \right),

\\ &
= \prod_{k=i}^{j-1} \text{Exp} \left( (\tilde{\omega}_k - b_{g,i}) \Delta t \right) \text{Exp} \left( -J_{r,k} \delta b_{g,i} \Delta t \right),

\\ &
= \text{Exp} \left( (\tilde{\omega}_i - b_{g,i}) \Delta t \right) \text{Exp} \left( -J_{r,i} \delta b_{g,i} \Delta t \right) \text{Exp} \left( (\tilde{\omega}_{i+1} - b_{g,i}) \Delta t \right) \text{Exp} \left( -J_{r,i+1} \delta b_{g,i} \Delta t \right) \cdots

\\ &
= \Delta \tilde{R}_{i,i+1} \text{Exp} \left( -J_{r,i} \delta b_{g,i} \Delta t \right) \Delta \tilde{R}_{i+1,i+2} \text{Exp}\left( -J_{r,i+1} \delta b_{g,i} \Delta t \right) \dots,

\\ &
= \Delta \tilde{R}_{i,i+1} \Delta \tilde{R}_{i+1,i+2} \text{Exp} \left( - \tilde{R}_{i+1,i+2}^\top J_{r,j} \delta b_{g,i} \Delta t \right) \dots,

\\ &
= \Delta \tilde{R}_{i,j} \prod_{k=i}^{j-1} \text{Exp} \left( -\Delta \tilde{R}_{k+1,j}^\top J_{r,k} \delta b_{g,i} \Delta t \right),

\\ &
\approx \Delta \tilde{R}_{i,j} \text{Exp} \left( -\sum_{k=i}^{j-1} \Delta \tilde{R}_{k+1,j}^\top J_{r,k} \Delta t \delta b_{g,i} \right).
\end{aligned}
\end{gather*}
$$

THe last step makes use of the fact that when angles are small, Jacobian $J \approx I $. So, multiplying them all together is approx adding up the angles in $Exp()$ So this gives the general Jacobian of the rotation part w.r.t gyro bias:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \tilde{\Delta R_{i,j}}}{\partial b_{g,i}} = \text{Exp} \left( -\sum_{k=i}^{j-1} \Delta \tilde{R}_{k+1,j}^\top J_{r,k} \Delta t \right)

\\ &
\text{Where inside the exponent:}

\\ &
-\sum_{k=i}^{j-1} \Delta \tilde{R}_{k+1,j}^\top J_{r,k} \Delta t

\\ &
= - \sum_{k=i}^{j-2} \Delta \tilde{R}_{k+1,j}^{\top} J_{r,k} \Delta t - \Delta \tilde{R}_{j,j}^{\top} J_{r,j-1} \Delta t,

\\ &
= - \sum_{k=i}^{j-2} \left( \Delta \tilde{R}_{k+1,j-1} \Delta \tilde{R}_{j-1,j} \right)^{\top} J_{r,k} \Delta t - J_{r,j-1} \Delta t,

\end{aligned}
\end{gather*}
$$

Written recursively:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}} =
\Delta \tilde{R}_{j-1,j}^{\top} \frac{\partial \Delta \tilde{R}_{i,j-1}}{\partial b_{g,i}} - J_{r,k} \Delta t.
\end{aligned}
\end{gather*}
$$

### Jacobian of Velocity Part w.r.t Gyro Bias And Accelerometer Bias

$$
\begin{gather*}
\begin{aligned}
& \Delta \tilde{v}_{ij} (b_{g,i} + \delta b_{g,i}, \mathbf{b}_{a,i} + \delta b_{a,i}) :=
\Delta \tilde{v}_{ij} (b_{g,i}, \mathbf{b}_{a,i}) + \frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{g,i}} \delta b_{g,i} + \frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{a,i}} \delta b_{a,i},

\\ &
\rightarrow

\\ &
= \Delta \tilde{v}_{ij} (b_i + \delta b_i) =
\sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} (b_{g,i} + \delta b_{g,i}) (\tilde{a}_k - b_{a,i} - \delta b_{a,i}) \Delta t,

\\ &
= \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} \text{Exp} \left( \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \delta b_{g,i} \right) (\tilde{a}_k - b_{a,i} - \delta b_{a,i}) \Delta t,

\\ &
\approx \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} \left( I + \left( \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \delta b_{g,i} \right)^{\wedge} \right) (\tilde{a}_k - b_{a,i} - \delta b_{a,i}) \Delta t,

\\ &
\approx \Delta \tilde{v}_{ij} - \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} \Delta t \delta b_{a,i} - \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} (\tilde{a}_k - b_{a,i})^{\wedge} \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \Delta t \delta b_{g,i},

\\ &
= \Delta \tilde{v}_{ij} + \frac{\partial \Delta v_{ij}}{\partial b_{a,i}} \delta b_{a,i} + \frac{\partial \Delta v_{ij}}{\partial b_{g,i}} \delta b_{g,i}.

\end{aligned}
\end{gather*}
$$

So, the velocity Jacobian is:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{a,i}} =
- \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} \Delta t,

\\ &
\frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{g,i}} =
- \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} (\tilde{a}_k - b_{a,i})^{\wedge} \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \Delta t.
\end{aligned}
\end{gather*}
$$


Written recursively:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{a,i}} =
\frac{\partial \Delta \tilde{v}_{i,j-1}}{\partial b_{a,i}} - \Delta \tilde{R}_{i,j-1} \Delta t,

\\ &
\frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{g,i}} =
\frac{\partial \Delta \tilde{v}_{i,j-1}}{\partial b_{g,i}} - \Delta \tilde{R}_{i,j-1} (\tilde{a}_{j-1} - b_{a,i})^{\wedge} \frac{\partial \Delta \tilde{R}_{i,j-1}}{\partial b_{g,i}} \Delta t.
\end{aligned}
\end{gather*}
$$

### Jacobian of Position Part w.r.t Gyro Bias And Accelerometer Bias

$$
\begin{gather*}
\begin{aligned}
& \Delta \tilde{p}_{ij} (b_{g,i} + \delta b_{g,i}, \mathbf{b}_{a,i} + \delta b_{a,i}) =
\Delta \tilde{p}_{ij} (b_{g,i}, \mathbf{b}_{a,i}) + \frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{g,i}} \delta b_{g,i} + \frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{a,i}} \delta b_{a,i}.

\\ &
\rightarrow

\\ &


\Delta \tilde{p}_{ij} (b_i + \delta b_i) \approx 
\sum_{k=i}^{j-1} \left[ \left( \Delta \tilde{v}_{ik} + \frac{\partial \Delta v_{ik}}{\partial b_{a,i}} \delta b_{a,i} + \frac{\partial \Delta v_{ik}}{\partial b_{g,i}} \delta b_{g,i} \right) \Delta t + \right.

\\ &
\left. \frac{1}{2} \Delta \tilde{R}_{ik} \left( I + \left( \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \delta b_{g,i} \right)^{\wedge} \right) (\tilde{a}_k - b_{a,i} - \delta b_{a,i}) \Delta t^2 \right],

\\ &
\approx \Delta \tilde{p}_{ij} + \sum_{k=i}^{j-1} \left[ \frac{\partial \Delta v_{ik}}{\partial b_{a,i}} \Delta t - \frac{1}{2} \Delta \tilde{R}_{ik} \Delta t^2 \right] \delta b_{a,i} +

\\ &
\sum_{k=i}^{j-1} \left[ \frac{\partial \Delta v_{ik}}{\partial b_{g,i}} \Delta t - \frac{1}{2} \Delta \tilde{R}_{ik} (\tilde{a}_k - b_{a,i})^{\wedge} \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \Delta t^2 \right] \delta b_{g,i},

\\ &
= \Delta \tilde{p}_{ij} + \frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{a,i}} \delta b_{a,i} + \frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{g,i}} \delta b_{g,i}.

\end{aligned}
\end{gather*}
$$

So the Jacobians of position part w.r.t. Gyro Bias and Accelerometer Bias are:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{a,i}} =
\sum_{k=i}^{j-1} \left[ \frac{\partial \Delta v_{ik}}{\partial b_{a,i}} \Delta t - \frac{1}{2} \Delta \tilde{R}_{ik} \Delta t^2 \right],

\\ &
\frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{g,i}} =
\sum_{k=i}^{j-1} \left[ \frac{\partial \Delta v_{ik}}{\partial b_{g,i}} \Delta t - \frac{1}{2} \Delta \tilde{R}_{ik} (\tilde{a}_k - b_{a,i})^{\wedge} \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \Delta t^2 \right].
\end{aligned}
\end{gather*}
$$

Written recursively:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{a,i}} =
\frac{\partial \Delta \tilde{p}_{i,j-1}}{\partial b_{a,i}} + \frac{\partial \Delta \tilde{v}_{i,j-1}}{\partial b_{a,i}} \Delta t - \frac{1}{2} \Delta \tilde{R}_{i,j-1} \Delta t^2,

\\ &
\frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{g,i}} =
\frac{\partial \Delta \tilde{p}_{i,j-1}}{\partial b_{g,i}} + \frac{\partial \Delta \tilde{v}_{i,j-1}}{\partial b_{g,i}} \Delta t - \frac{1}{2} \Delta \tilde{R}_{i,j-1} (\tilde{a}_{j-1} - b_{a,i})^{\wedge} \frac{\partial \Delta \tilde{R}_{i,j-1}}{\partial b_{g,i}} \Delta t^2.
\end{aligned}
\end{gather*}
$$