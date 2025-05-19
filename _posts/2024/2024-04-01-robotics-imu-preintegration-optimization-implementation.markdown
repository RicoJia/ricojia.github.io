---
layout: post
title: Robotics - IMU Pre-integration Optimization Implementation
date: '2024-04-01 13:19'
subtitle: IMU Pre-Integration, GNSS, UTM
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## G2O Recap - What's An Edge

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
    2. Jacobian w.r.t $p_i$: ` dp/dp1 = -R1T.matrix();`

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

### Acceleration Bias Estimates

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


