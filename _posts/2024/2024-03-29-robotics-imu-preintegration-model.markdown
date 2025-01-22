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
v_j = v_i +g_k \Delta t_{ij} + \sum_{k=i}^{j-1} R_k (\tilde{a_k} - b_{a,k} - \eta{ad, k}) \Delta t
\\ &
p_j = p_i + \sum_{k=i}^{j-1} v_k \Delta t + \frac{1}{2} g_k \Delta t_{ij}^2 + \frac{1}{2} \sum_{k=i}^{j-1} R_k (\tilde{a_k} - b_{a,k} - \eta{ad, k}) \Delta t^2 
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
\Delta v_{ij} := R_i^T(v_j - v_i - g_k \Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta R_{ik} (\tilde{a_k} - b_{a,k} - \eta{ad, k}) \Delta t
\\ &
\Delta p_{ij} := R_i^T(p_j - p_i - v_i \Delta t_{ij} - \frac{1}{2} g_k \Delta t_{ij}^2) = \sum_{k=i}^{j-1} \Delta v_{ik} \Delta t + \frac{1}{2} \sum_{k=i}^{j-1} \Delta R_{ik} (\tilde{a_k} - b_{a,k} - \eta{ad, k}) \Delta t^2 
\end{aligned}
\end{gather*}
$$

- The "rotation part" $\Delta R_{ij}$ is the accumulated rotation between i, j
- The "velocity part" $\Delta v_{ij}$ and the "position part" $\Delta p_{ij}$ are less intuitive. But all three values start at 0 at ith time. They are in the forms of product or sum, which makes later linearization with Jacobian easier.
    - With linearization, we can just apply correction terms based if bias terms like $b_{a} changes.
- All three values are independent of absolute state variables

## Pre-integration Model