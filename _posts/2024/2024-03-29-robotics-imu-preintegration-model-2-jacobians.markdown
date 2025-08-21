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

## Using Pre-Integration Terms as Edges in Graph Optimization

Formulating a graph optimization problem using pre-integration (as edges) and states as nodes is quite flexible. [Recall that from here](https://ricojia.github.io/2024/07/11/rgbd-slam-bundle-adjustment/) that a graph optimization problem is formulated as:

$$
\begin{gather*}
\begin{aligned}
& F(X) = \sum_{i \leq 6, j \leq 6} (r_{ij})^T \Omega r_{ij}

\\ &
\text{Approximating F(x) to find its minimum more easily:}

F(X + \Delta X) = e(X+\Delta X)^T \Omega e(X+\Delta X)^T
\\ &
\approx (e(X) + J \Delta X)^T \Omega (e(X) + J \Delta X)
\\ &
= C + 2b \Delta X + \Delta X^T H \Delta X

\\ &
\text{Where:}

\\ &
J = \frac{\partial r_{ij}}{\partial(X)}

\\ &
H = \sum_{ij} H_{ij} = \sum_{ij} J^T_{ij} \Omega J_{ij} \text{(Gauss Newton)}
\\ &
\text{OR}
\\ &
H = \sum_{ij} H_{ij} = \sum_{ij} (J^T_{ij} \Omega J_{ij} + \lambda I) \text{(Levenberg-Marquardt)}
\end{aligned}
\end{gather*}
$$

One way is to use a single node to encompass all states. That however, would create a giant Jacobian & Hessian for the problem, but in the meantime there are a lot of zeros in them. So now we use separate nodes for each state.

**The error, a.k.a residual**, can be defined flexibly as well. Here, we define it to be the `difference` between the integration terms calculated from our state estimates, and the ones come from our IMU (but with $b_g$ and $b_a$).

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/474a4208-f8d7-418e-ad3a-bbad92478217" height="300" alt=""/>
        <figcaption>Source: 深蓝学院</figcaption>
    </figure>
</p>
</div>

So formally, we define our residuals to be:

$$
\begin{gather*}
\begin{aligned}
& r_{\Delta R_{ij}} = \log \left( \Delta \tilde{R}_{ij}^{\top} \left( R_i^{\top} R_j \right) \right),
\\ &
r_{\Delta v_{ij}} = R_i^{\top} \left( v_j - v_i - g \Delta t_{ij} \right) - \Delta \tilde{v}_{ij}
\\ &
r_{\Delta p_{ij}} = R_i^{\top} \left( p_j - p_i - v_i \Delta t_{ij} - \frac{1}{2} g \Delta t_{ij}^2 \right) - \Delta \tilde{p}_{ij}.
\end{aligned}
\end{gather*}
$$

Now, the question is: what's the Jacobian of each residual, with respect to each element?

### Jacobian of the Rotation Part w.r.t Angles

To find the Jacobian, (the first order partial derivatives), we can go back to the original definition of derivative:

$$
\begin{gather*}
\begin{aligned}
& J = \lim_{\phi \to 0} \frac{r_{\Delta R_{ij}}(R(\phi)) - r_{\Delta R_{ij}}(R)}{\phi}
\end{aligned}
\end{gather*}
$$

Where $R(\phi)$ represents the perturbed rotation.

By using [this property](https://ricojia.github.io/2017/02/22/lie-group/#3-rt-textexpphi-r--textexprt-phi) and the BCH formula, we can write out the right perturbation of $\phi_i$

$$
\begin{gather*}
\begin{aligned}
& \begin{aligned}
r_{\Delta R_{ij}}(R_i \operatorname{Exp}(\phi_i)) &= \log \left( \Delta \tilde{R}_{ij} \left( (R_i \operatorname{Exp}(\phi_i))^{\top} R_j \right) \right), \\
&= \log \left( \Delta \tilde{R}_{ij} \operatorname{Exp}(-\phi_i) R_i^{\top} R_j \right), \\
&= \log \left( \Delta \tilde{R}_{ij} R_i^{\top} R_j \operatorname{Exp}(-R_i^{\top} R_j \phi_i) \right), \\
&= r_{\Delta R_{ij}} - J_r^{-1}(r_{\Delta R_{ij}}) R_j^{\top} R_i \phi_i.
\end{aligned}

\end{aligned}
\end{gather*}
$$

The perturbation of $\phi_j$:

$$
\begin{gather*}
\begin{aligned}
r_{\Delta R_{ij}}(R_j \operatorname{Exp}(\phi_j)) &= \log \left( \Delta \tilde{R}_{ij} R_i^{\top} R_j \operatorname{Exp}(\phi_j) \right), \\
&= r_{\Delta R_{ij}} + J_r^{-1}(r_{\Delta R_{ij}}) \phi_j.
\end{aligned}
\end{gather*}
$$

So the Jacobians are:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial r_{\Delta R_{ij}}}{\partial \phi_i} = - J_r^{-1}(r_{\Delta R_{ij}}) R_j^{\top} R_i

\\ &
\frac{\partial r_{\Delta R_{ij}}}{\partial \phi_j} = J_r^{-1}(r_{\Delta R_{ij}})
\end{aligned}
\end{gather*}
$$

Meanwhile, the rotational error is a function of gyro bias $b_g$ as well. In an arbitrary iteration, we calculate a correction $\delta b_g$. When calculating the Jacobian (for the next iteration update), we need to take that into account as well:

$$
\begin{gather*}
\begin{aligned}
r_{\Delta R_{ij}}(b_{g,i} + \delta b_{g,i} + \tilde{\delta} b_{g,i}) &= \log \left( \left( \Delta \tilde{R}_{ij} \operatorname{Exp} \left( \frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}} (\delta b_{g,i} + \tilde{\delta} b_{g,i}) \right) \right)^{\top} R_i^{\top} R_j \right), \\
&\overset{\text{BCH}}{\approx} \log \left( \left( \underbrace{\Delta \tilde{R}_{ij} \operatorname{Exp} \left( \frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}} \delta b_{g,i} \right) \operatorname{Exp} \left( J_{r,b} \frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}} \tilde{\delta} b_{g,i} \right)}_{\tilde{R}_{ij}'} \right)^{\top} R_i^{\top} R_j \right), \\
&= \log \left( \operatorname{Exp} \left( - J_{r,b} \frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}} \tilde{\delta} b_{g,i} \right) (\Delta \tilde{R}_{ij}')^{\top} R_i^{\top} R_j \right), \\
&= \log \left( \operatorname{Exp} \left( r'_{\Delta R_{ij}} \right) \operatorname{Exp} \left( - \operatorname{Exp} \left( r'_{\Delta R_{ij}} \right)^{\top} J_{r,b} \frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}} \tilde{\delta} b_{g,i} \right) \right), \\
&\approx r'_{\Delta R_{ij}} - J_r^{-1} (r'_{\Delta R_{ij}})^{\top} J_{r,b} \frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}} \tilde{\delta} b_{g,i}.
\end{aligned}
\end{gather*}
$$

So the partial derivative of the rotational part w.r.t gyro bias $b_g$ is:

$$
\begin{gather*}
\begin{aligned}
\frac{\partial r_{\Delta R_{ij}}}{\partial b_i} = - J_r^{-1} (r'_{\Delta R_{ij}})^{\top} J_{r,b} \frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}}
\end{aligned}
\end{gather*}
$$

### Jacobians of the Velocity Part

Since:

$$
\begin{gather*}
\begin{aligned}
& r_{\Delta v_{ij}} = R_i^{\top} \left( v_j - v_i - g \Delta t_{ij} \right) - \Delta \tilde{v}_{ij}
\end{aligned}
\end{gather*}
$$

The Jacobians w.r.t to $v_i$, $v_j$ are very intuitive,

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial r_{\Delta v_{i,j}}}{\partial v_i} = -R_i^T
\\ &
\frac{\partial r_{\Delta v_{i,j}}}{\partial v_j} = R_i^T
\end{aligned}
\end{gather*}
$$

For rotation, we simply use first order expansion:

$$
\begin{gather*}
\begin{aligned}
r_{\Delta v_{ij}} \left( R_i \operatorname{Exp}(\delta \phi_i) \right) &= \left( R_i \operatorname{Exp}(\delta \phi_i) \right)^{\top} \left( v_j - v_i - g \Delta t_{ij} \right) - \Delta \tilde{v}_{ij}, \\
&= \left( I - \delta \phi_i^{\wedge} \right) R_i^{\top} \left( v_j - v_i - g \Delta t_{ij} \right) - \Delta \tilde{v}_{ij}, \\
&= r_{\Delta v_{ij}} \left( R_i \right) + \left( R_i^{\top} \left( v_j - v_i - g \Delta t_{ij} \right) \right)^{\wedge} \delta \phi_i.
\end{aligned}
\end{gather*}
$$

For velocity, recall that the Jacobian of the "observed" velocity part w.r.t biases are:

$$
\begin{gather*}
\begin{aligned}
\frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{a,i}} &= - \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} \Delta t, \\
\frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{g,i}} &= - \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} \left( \tilde{a}_k - b_{a,i} \right)^{\wedge} \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \Delta t.
\end{aligned}
\end{gather*}
$$

Since in $r_{\Delta v_{ij}}$, only the $-\Delta \tilde{v}_{ij}$ is a function of the biases,

$$
\begin{gather*}
\begin{aligned}
& r_{\Delta v_{ij}} = R_i^{\top} \left( v_j - v_i - g \Delta t_{ij} \right) - \Delta \tilde{v}_{ij}
\end{aligned}
\end{gather*}
$$

We can get:

$$
\begin{gather*}
\begin{aligned}
\frac{\partial r_{\Delta v_{i,j}}}{\partial b_{a,i}} &= \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} \Delta t, \\
\frac{\partial r_{\Delta v_{i,j}}}{\partial b_{g,i}} &= \sum_{k=i}^{j-1} \Delta \tilde{R}_{ik} \left( \tilde{a}_k - b_{a,i} \right)^{\wedge} \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \Delta t.
\end{aligned}
\end{gather*}
$$

### Jacobians of the Position Part

Using First order taylor expansion, it's easy to get:

$$
\begin{gather*}
\begin{aligned}
\frac{\partial r_{\Delta p_{ij}}}{\partial p_i} &= - R_i^{\top}, \\
\frac{\partial r_{\Delta p_{ij}}}{\partial p_j} &= R_i^{\top}, \\
\frac{\partial r_{\Delta p_{ij}}}{\partial v_i} &= - R_i^{\top} \Delta t_{ij}, \\
\frac{\partial r_{\Delta p_{ij}}}{\partial \phi_i} &= \left( R_i^{\top} \left( p_j - p_i - v_i \Delta t_{ij} - \frac{1}{2} g \Delta t_{ij}^2 \right) \right)^{\wedge}.
\end{aligned}
\end{gather*}
$$

And for the biases, we simply reverse the signs just like the velocity part:

$$
\begin{gather*}
\begin{aligned}
\frac{\partial r_{\Delta p_{i,j}}}{\partial b_{a,i}} &= -\sum_{k=i}^{j-1} \left[ \frac{\partial \Delta \tilde{v}_{ik}}{\partial b_{a,i}} \Delta t - \frac{1}{2} \Delta \tilde{R}_{ik} \Delta t^2 \right], \\
\frac{\partial r_{\Delta p_{i,j}}}{\partial b_{g,i}} &= -\sum_{k=i}^{j-1} \left[ \frac{\partial \Delta \tilde{v}_{ik}}{\partial b_{g,i}} \Delta t - \frac{1}{2} \Delta \tilde{R}_{ik} \left( \tilde{a}_k - b_{a,i} \right)^{\wedge} \frac{\partial \Delta \tilde{R}_{ik}}{\partial b_{g,i}} \Delta t^2 \right].
\end{aligned}
\end{gather*}
$$

### Formulation

In a graph optimization systems, we have keyframes.

1. Given the current estimates of biases, we can adjust the pre-integration in a linear manner.
1. The residuals are edges (constraints) between nodes.

When a new IMU data comes in:

1. Calculate $\Delta R_{ij}, \Delta v_{ij}, \Delta p_{ij}$
1. Calculate noise covariances as the information matrices for the graph optimization
1. Jacobians of Pre-integration w.r.t biases (so we can update them in a linear manner): $\frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}},
\frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{a,i}},
\frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{g,i}},
\frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{a,i}},
\frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{g,i}}$
