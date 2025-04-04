---
layout: post
title: Robotics - [3D SLAM - 1] Scan Matching
date: '2025-2-1 13:19'
subtitle: 3D Point-Point ICP, 3D Point-Plane ICP, NDT
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - SLAM
comments: true
---

## Point-Point ICP

If we write out our state vector as `[theta, translation]`, the pseudo code for point-point ICP is:

```
pt_pt_icp(pose_estimate, source_scan, target_scan):
    build_kd_tree(target_scan)
    for n in ITERATION_NUM:
        for pt in source_scan:
            pt_map = pose_estimate * pt
            pt_map_match = kd_tree_nearest_neighbor_search(pt_map)
            errors[pt] = pt_map_match - pt_map
            jacobians[pt] = [pose_estimate.rotation() * hat(pt_map), -Identity(3)]
        total_residual = errors* errors
        H = sum(jacobians.transpose() * jacobians)
        b = sum(-jacobians * errors)
        dx = H.inverse() * b
       pose_estimate += dx;
        if get_total_error() -> constant:
            return
```

Note that in Point-Point ICP, for scan matched point (in map frame) `p_i` and its matched point `q_i`, if pose estimate is decomposed into `[R, t]`, we set:

$$
\begin{gather*}
\begin{aligned}
& e = q_i - Rp_i - t
\\ &
\frac{\partial e}{\partial R} = R[p_i]^\land, \frac{\partial e}{\partial t} = -I
\end{aligned}
\end{gather*}
$$

### Notes

- Iterative Method is not a must. Unlike the above point association -> pose estimate -> ... iteration, recent scan matching methods solve point association and pose estimate together. TODO?
    - Iterative methods are relatively simpler, though. **It works as long as the cost landscape is decreasing from the current estimate to the global minimum.**

## Point-Plane ICP

If we write out our state vector as `[theta, translation]`, the pseudo code for point-plane ICP is:

```
pt_plane_icp(pose_estimate, source_scan, target_scan):
    build_kd_tree(target_scan)
    for n in ITERATION_NUM:
        for pt in source_scan:
            pt_map = pose_estimate * pt
            N_pt_map_matches = kd_tree_nearest_neighbor_search(N, pt_map)
            plane_coeffs = fit_plane(N_pt_map_matches)
            errors[pt] = signed_distance_to_plane(plane_coeffs, pt_map)
            jacobians[pt] = get_jacbian(plane_coeffs, pt_map)
        total_residual = errors* errors
        H = sum(jacobians.transpose() * jacobians)
        b = sum(-jacobians * errors)
        dx = H.inverse() * b
       pose_estimate += dx;
        if get_total_error() -> constant:
            return
```

The Point-Plane ICP is different from Point-Point ICP in:
- We are trying to fit a plane instead of using point-point match
- In real life, we need a distance threshold for filtering out distance outliers.


Also note that:
- A plane is $n^T x + d = 0$. The plane coefficient we get are $n, d$
- The signed distance between a point and a plane is:

$$
\begin{gather*}
\begin{aligned}
& e_i = n^T(R q_i + t) + d
\end{aligned}
\end{gather*}
$$

- The jacobian of error is:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial e_i}{\partial R} = -n^T R q^\land
\\ & 
\frac{\partial e_i}{\partial t} = n^T
\end{aligned}
\end{gather*}
$$

## Point-Line ICP

The point-line ICP is mostly similar to the point-plane ICP, except that we need to adjust the error and Jacobians. 

A line is:

$$
\begin{gather*}
\begin{aligned}
& d \vec{r} + p_0 
\end{aligned}
\end{gather*}
$$

We choose the error as the signed distance between a point and it is : 

$$
\begin{gather*}
\begin{aligned}
& e = \vec{r} \times (R p_t + t - p_0)
\\ &
= \vec{r} ^ \land (R p_t + t - p_0)
\end{aligned}
\end{gather*}
$$

The jacobian is:

$$
\begin{gather*}
\begin{aligned}
& \frac{\partial e}{\partial R} = - \vec{r} ^\land R p_t^{\land}
\\ 
& \frac{\partial e}{\partial t} = \vec{r} ^\land
\end{aligned}
\end{gather*}
$$

## NDT (Normal Distribution Transform)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/39393023/429244689-5b57d20e-5652-4541-9bf4-1231d7fc4ee0.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250401%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250401T225620Z&X-Amz-Expires=300&X-Amz-Signature=1eaa6634725d7bdc4025d581196115dde9b0f911119d554cc88663e0ad1b35d7&X-Amz-SignedHeaders=host" height="300" alt=""/>
        <figcaption><a href="https://blog.csdn.net/jinshengtao/article/details/103828230">Source: CSDN </a></figcaption>
    </figure>
</p>
</div>


1. Voxelization. For each voxel, calculate the mean and variance of its points: $\mu$, $\Sigma$
2. For each point in the source scan:
    1. Calculate its map pose $pt$
    2. Calculate the voxel of $pt$. Grab all points in the same voxel
    3. We believe that if the source is well-aligned to the target, the distribution of the points in the same voxel is the same as that of the target. So: 
        1. $e_i = Rp_t + t - \mu$
        2. $(R,t) = \text{argmin}_{R,t} [\sum e_i^t \Sigma^{-1} e_i]$
            - This is equivalent to Maximum Likelihood Estimate (MLE)
            - $\text{argmax}_{R,t} [\sum log(P(R q_i + t))]$
        3. Jacobians:
            - $\frac{\partial e_i}{\partial R} = -Rp_t^\land$
            - $\frac{\partial e_i}{\partial t} = I$

This is more similar to the 2D version of NDT [2], and it different from the original paper [1]. But the underlying core is the same: in 2009, SE(3) manifold optimization was not popular, therefore the original paper used sin and cos to represent the derivative of the cost function. In reality, modifications / simplifications like this are common. 

**Another consideration is that we add a small positive value to the covariance matrix**, because we need to get its inverse. When points happen to be on a line or a plane, elements [on its least principal vectors will become zero.](https://ricojia.github.io/2017/01/15/eigen-value-decomp/)

## Comparison TODO

- PCL is slower, why?

    <div style="text-align: center;">
        <p align="center">
        <figure>
                <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/39393023/428796840-64e87f07-dfa5-40c0-9bc5-840f3cea0ae6.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250331%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250331T220659Z&X-Amz-Expires=300&X-Amz-Signature=6d211d366a886a3715af956062f0011614de52b2301fa7757738143a66ee0294&X-Amz-SignedHeaders=host" height="300" alt=""/>
        </figure>
        </p>
    </div>

## References

[1] M. Magnusson, The three-dimensional normal-distributions transform: an efficient representation for registra-
tion, surface analysis, and loop detection. PhD thesis, Örebro universitet, 2009.

[2] P. Biber and W. Straßer, “The normal distributions transform: A new approach to laser scan matching,” in
Proceedings 2003 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2003)(Cat. No.
03CH37453), vol. 3, pp. 2743–2748, IEEE, 2003.