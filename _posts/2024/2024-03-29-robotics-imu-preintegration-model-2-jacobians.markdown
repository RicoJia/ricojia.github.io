---
layout: post
title: Robotics - [IMU Pre-integration Model 2] How To Update Pre-integration With Changing IMU Biases
date: '2024-03-29 13:19'
subtitle: Jacobian Derivation
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

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

The last step makes use of the fact that when angles are small, Jacobian $J \approx I $. So, multiplying them all together is approx adding up the angles in $Exp()$ So this gives the general Jacobian of the rotation part w.r.t gyro bias:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial \tilde{\Delta R_{i,j}}}{\partial b_{g,i}} =  -\sum_{k=i}^{j-1} \Delta \tilde{R}_{k+1,j}^\top J_{r,k} \Delta t

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
\begin{aligned}
\Delta\tilde{p}_{ij}\!\left(b_{g,i}+\delta b_{g,i},\, b_{a,i}+\delta b_{a,i}\right)
&= \Delta\tilde{p}_{ij}(b_{g,i}, b_{a,i})
 + \frac{\partial \Delta\tilde{p}_{ij}}{\partial b_{g,i}}\,\delta b_{g,i}
 + \frac{\partial \Delta\tilde{p}_{ij}}{\partial b_{a,i}}\,\delta b_{a,i}
\\[6pt]
&\Longrightarrow
\\[4pt]
\Delta\tilde{p}_{ij}(b_i+\delta b_i)
&\approx \sum_{k=i}^{j-1} \Bigl[
\bigl(\Delta\tilde{v}_{ik}
   + \tfrac{\partial \Delta\tilde{v}_{ik}}{\partial b_{a,i}}\,\delta b_{a,i}
   + \tfrac{\partial \Delta\tilde{v}_{ik}}{\partial b_{g,i}}\,\delta b_{g,i}
 \bigr)\Delta t
\\[-2pt]
&\qquad\quad
+ \tfrac{1}{2}\,\Delta\tilde{R}_{ik}
\Bigl(I + \bigl(\tfrac{\partial \Delta\tilde{R}_{ik}}{\partial b_{g,i}}\,\delta b_{g,i}\bigr)^{\wedge}\Bigr)
\bigl(\tilde{a}_k - b_{a,i} - \delta b_{a,i}\bigr)\,\Delta t^{2}
\Bigr]
\\[6pt]
&\approx \Delta\tilde{p}_{ij}
+ \sum_{k=i}^{j-1} \Bigl[
\frac{\partial \Delta\tilde{v}_{ik}}{\partial b_{a,i}}\,\Delta t
- \tfrac{1}{2}\,\Delta\tilde{R}_{ik}\,\Delta t^{2}
\Bigr]\delta b_{a,i}
\\
&\quad
+ \sum_{k=i}^{j-1} \Bigl[
\frac{\partial \Delta\tilde{v}_{ik}}{\partial b_{g,i}}\,\Delta t
- \tfrac{1}{2}\,\Delta\tilde{R}_{ik}\,(\tilde{a}_k - b_{a,i})^{\wedge}\,
\frac{\partial \Delta\tilde{R}_{ik}}{\partial b_{g,i}}\,\Delta t^{2}
\Bigr]\delta b_{g,i}
\\[4pt]
&= \Delta\tilde{p}_{ij}
  + \frac{\partial \Delta\tilde{p}_{ij}}{\partial b_{a,i}}\,\delta b_{a,i}
  + \frac{\partial \Delta\tilde{p}_{ij}}{\partial b_{g,i}}\,\delta b_{g,i}.
\end{aligned}
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
