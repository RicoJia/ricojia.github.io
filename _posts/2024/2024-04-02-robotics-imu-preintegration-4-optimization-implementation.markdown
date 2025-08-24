---
layout: post
title: Robotics - [IMU Pre-integration Model 4] IMU Pre-integration Optimization Implementation
date: '2024-04-02 13:19'
subtitle: IMU Pre-Integration, GNSS, UTM
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

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
1. Jacobians of Pre-integration w.r.t biases (so we can update them in a linear manner):

$$
\begin{gather*}
\begin{aligned}
\frac{\partial \Delta \tilde{R}_{ij}}{\partial b_{g,i}}, \quad
\frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{a,i}}, \quad
\frac{\partial \Delta \tilde{v}_{ij}}{\partial b_{g,i}}, \quad
\frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{a,i}}, \quad
\frac{\partial \Delta \tilde{p}_{ij}}{\partial b_{g,i}}.
\end{aligned}
\end{gather*}
$$

## G2O Recap

A single edge is a "difference" between two conceptual entities. The entities are conceptual because they don't need to map exactly to your implementation. For example, for certain reasons, we choose to set `x` and `y` coordinates as separate vertices. But now if we want to calculate the difference between two car poses using Euler Distance, conceptually, the edge represents `dist(p2-p1)`, but the implementation will be:

```
L2(x1^2 + y1^2 - x2^2 - y2^2)
```

In G2O, an edge is a hyper edge. That is, you can congregate multiple "differences" together as a single "hyper difference", with multiple vertices connecting to it. E.g., to compute the full pose difference, we can congregate Euler Distance and Angular Distance into a single vector together.

```
[Euler_diff, angular diff]
```

This formulation helps us vectorize error computation, so we can update them with linear updates all together.

## Formulations

### Estimate Formulation

We formulate a simple two-frame `g2o` optimization framework based on the pre-integration trick and the associated Jacobian. Assume we have received two GNSS measurements, we put the measurements into GNSS frames A and B. Each frame has an estimate of: orientation, (x,y,z), velocity, gyro bias, and acceleration bias.

Frame is simply a datastructure that stores estimates of these variables. They need to be loaded into vertices for any optimization to start.

### Orientation Estimates

When Frame A is first created, it gets an estimate from the GNSS. During Optimization. R is updated by these constraints (edges):

- IMU Pre-integration Edge (EdgeInertial):
    1. Given the current gyro estimate, get the pre-integration rotation part: `const SO3 dR = preint_->GetDeltaRotation(bg);`
    2. Calculate the residual as error: $eR = dR^{-1} * R_i^T * R_j$
    3. For Update, the Jacobian of the residual w.r.t orientation update is: $J_r^{-1} (R_j^{-1} R_i)$
- Note that we do update R in a pose estimate in Update_Constraint Edge (EdgePriorPoseNavState, based on the prior)
    1. `const Vec3d er = SO3(state_.R_.matrix().transpose() * vp->estimate().so3().matrix()).log();` for the orientation update. Its Jacobian is:

        ```cpp
        const Vec3d er = SO3(state_.R_.matrix().transpose() * vp->estimate().so3().matrix()).log();
        _jacobianOplus[0].block<3, 3>(0, 0) = SO3::jr_inv(er);    // dr/dr
        ```

### Velocity Estimates

Each frame has a velocity estimate that starts at 0. When frame A is the most up-to-date frame, it does not get updated during the observation stage. It is updated during the optimization stage:

- Odom edge (`EdgeEncoder3D`)
  - The residual (a.k.a error) is simply `(velocity_estimate - v_linear_odom)`. `velocity_estimate` is the only vertex
  - For Update, the Jacobian of the residual w.r.t velocity_estimate is Identity.
- IMU Pre-integration Edge
  - The residual is $r_{\Delta v_{i,j}} = R_i^T (v_j - v_i - g \Delta t_{i,j}) - \Delta \tilde{v}_{ij}$. So `velocity_estimate` of the current frame and the last frame are the vertices required
  - For Update, the Jacobian of the residual w.r.t velocity_estimate is $-R_i^T$ and $R_i^T$ for Frame A and Frame B respectively

At the next time, a new frame B is created. Frame A does not get updated during observation stage, but get updated udring the optimization stage:

- IMU Pre-integration Edge (see above)
- Update_Constraint Edge (EdgePriorPoseNavState, based on the prior)
  - Measures `vertex(velocity_estimate) - velocity_estimate` so the update on the vertex is not substantial. This edge has constraints on other variables to update: position, gyro bias, and acceleration bias.
  - For Update, the Jacobian of the residual w.r.t velocity_estimate is Identity.

### Position Estimates

At time step B, only frame A is used for these constraints:

- EdgeGNSS
  - Measures the difference in:
        1. `derror_r = state_rotation^T * GNSS_rotation`. This orientation update, but it's integrated conveniently into `SE(3)`
        2. `derror_p = state_position - GNSS_position`
  - Jacobian:
        1. `derror_r / d_{state_rotation} = (_measurement.so3().inverse() * v->estimate().so3()).jr_inv();  // dR/dR`
        2. `derror_p/d_{state_position} = I`

- Update_Constraint Edge (EdgePriorPoseNavState, based on the prior).  Position is used in :
    1. `const Vec3d ep = vp->estimate().translation() - state_.p_;`
    2. Jacobian w.r.t $p_i$: `dp/dp1 = -R1T.matrix();`

Frame B is used for:

- `IMU Pre-integration Edge`(EdgeInertial)
    1. Calculate the delta position part according to the IMU: `dp=preint_->GetDeltaPosition(bg, ba);`
    2. Calculate the difference between the IMU position and the position from current state estimates (residual)`const Vec3d ep = RiT * (p2->estimate().translation() - p1->estimate().translation() - v1->estimate() * dt_ - grav_ * dt_ * dt_ / 2) - dp;`

- EdgeGNSS (see above)

### Gyro Bias Estimates

At the first GNSS frame, the gyro bias is created with the bg IMU initialization. `options_.preinteg_options_.init_bg_`.Then, for all subsequent frames, it gets updated in the optimization process. In two given frames A and B, the bias estimates are updated in these edges:

- `EdgeGyroRW`
  - This edge is simply the difference between the gyro biases in two frames: `bg_j->estimate() - bg_i->estimate()`.
  - The Jacobian is very intutitve: `I` for `bg_j`, and `-I` for `bg_i`
- Update_Constraint Edge (EdgePriorPoseNavState, based on the prior)
  - There are multiple Jacobian of sub-residuals w.r.t `bg`. The sub-residuals include $dR$, $dP$, and $dV$
  - `dR/dbg1 = -invJr * eR.inverse().matrix() * SO3::jr((dR_dbg * dbg).eval()) * dR_dbg;` Note that above rotation residual `eR`  and jacobian of the rotation part w.r.t `bg` have been calculated
  - `dV/dbg1 = -dv_dbg`. `dv_dbg` is updated in pre-integration: `dV_dbg_ - dR_.matrix() * dt * acc_hat * dR_dbg_;`
  - `dP/dbg1 = -dp_dbg` is also updated in pre-integration: `dp_dbg = dP_dbg_ + dV_dbg_ * dt - 0.5f * dR_.matrix() * dt2 * acc_hat * dR_dbg_;`

## Program Architecture

Upon receiving a new GNSS or odom update:

1. Create a new frame object.
2. Get an estimate of the current frame by $lastframe \oplus imupreintegration$
3. Optimize
    - Add regular edges, like Prior edges, and pre-integration edges. Some examples include:
        - V: IMU Pre-integration Edge, between 2 GNSS frames

            ```
            r_{\Delta v_{i,j}} = R_i^T (v_j - v_i - g \Delta t_{i,j}) - \Delta \tilde{v}_{ij}
            - For Update, the Jacobian of the residual w.r.t velocity_estimate is $-R_i^T$ and $R_i^T$ for Frame A and Frame B respectively
            ```

        - R:
            - IMU Pre-integration Edge()

                ```
                const SO3 dR = preint_->GetDeltaRotation(bg);
                eR = dR^{-1} * R_i^T * R_j
                J_R
                ```

            - EdgePriorPoseNavState

                ```
                const Vec3d er = SO3(state_.R_.matrix().transpose() * vp->estimate().so3().matrix()).log();
                // Jacobian:
                const Vec3d er = SO3(state_.R_.matrix().transpose() * vp->estimate().so3().matrix()).log();
                _jacobianOplus[0].block<3, 3>(0, 0) = SO3::jr_inv(er);    // dr/dr
                ```

    - If this frame is a GNSS update: add an GNSS  edge
    - If this frame is an Odom update: add an odom edge

        ```
        (current_frame_velocity_estimate - v_linear_odom)`
        ```

4. Update the current frame

## GNSS

### UTM (Universal Transverse Mercator)

The key idea is to place the globe into a slightly smaller cylinder. The center line will later become the "meridian" of a zone. The projection of a small Latitudinal line will have a bit of distortion there. At the two touching points, (the standard lines), the distortion is zero, then it will increase (until the zone boundaries). Then, we rotate the globe, and repeat again . There are in total 60 zones

[See this video for visualization](https://youtu.be/cfrxauufID4)
