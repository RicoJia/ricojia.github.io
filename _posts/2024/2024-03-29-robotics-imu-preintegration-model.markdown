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
& \Delta R_{ij} := R_i^T R_j \prod_{k=i}^{j-1} (Exp((\tilde{w_k} - b_{g,k} - \eta_{gd, k})\Delta t))
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
= \Delta \tilde{R}_{i,j} \prod_{k=0}^{j-1}
Exp \left( -\Delta \tilde{R}_{i+k,i+k+1}^\top J_{r,i} \eta_{gd,i} \Delta t \right) \cdots

\\ &
= \Delta \tilde{R}_{i,j} Exp \left(-\delta \phi_{i,j} \right)
\end{aligned}
\end{gather*}
$$

Where the accumulated observed rotation part is $\Delta \tilde{R}_{i,j}$

#### I'm not sure about... TODO: 

- Is this similarity transformation??

$$
\begin{gather*}
\begin{aligned}
& Exp(A)R = RExp(R^TA)
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
\rightarrow = \Delta \tilde{v_{ij}} +  \sum_{k=i}^{j-1} \Delta \tilde{R}_{i,k} (\tilde{a_k} - b_{a,k} - \eta_{ad, k})^{\land} \phi_{i,k} \Delta t - \Delta \tilde{R}_{i,k} \eta_{ad, k}

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
& Exp \left(-\delta \phi_{i,j} \right) = \prod_{k=0}^{j-1}
Exp \left( -\Delta \tilde{R}_{i+k,i+k+1}^\top J_{r,i} \eta_{gd,i} \Delta t \right) 

\end{aligned}
\end{gather*}
$$

Now let's get 

$$
\begin{gather*}
\begin{aligned}
& \phi_{ij} = -Log(\prod_{k=0}^{j-1}
Exp \left( -\Delta \tilde{R}_{i+k,i+k+1}^\top J_{r,i} \eta_{gd,i} \Delta t \right))
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
& \phi_{ij} \approx \sum_{k=i}^{j} \Delta \tilde{R}_{i+k,i+k+1}^\top J_{r,i} \eta_{gd,i} \Delta t
\end{aligned}
\end{gather*}
$$

The mean is only a linear combination of with zero-mean gaussian noise $\eta_{gd,i}$, so the mean is zero. Now let's get covariance