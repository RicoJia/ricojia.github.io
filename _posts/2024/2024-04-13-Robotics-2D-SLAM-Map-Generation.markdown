---
layout: post
title: Robotics - [2D SLAM 2] Map Generation
date: '2024-04-13 13:19'
subtitle: Submap Generation
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
    - SLAM
---

## Submap Generation

In modern robotics mapping systems, efficient environment representation and robust localization are achieved by dividing the map into **smaller, manageable submaps**. Each submap is built upon two key components: an occupancy map that records obstacles and free space, and a likelihood field derived from the occupancy map. The workflow of submap generation is as follows:

1. Initialize the occupancy map with the first scan. 
2. Generate a likelihood field according to the scan
3. For subsequent scans:
    1. Perform Scan-match against the likelihood to find the pose estimate from the last scan `T_21`. 
    2. If `T_21` shows a translation or a rotation larger than their thresholds. The current scan is a **keyframe**. 
    3. Add the keyframe into the occupancy map. 
    4. Generate a likelihood field according to the scan
    5. Create a new submap with its origin being the current scan's origin, if any below condition is met:
        1. The occupancy map already has `N` submaps
        2. Any point already falls outside of the map boundary. 

## Loop Closure

### Multi-resolution Scan Matching

1. Generate M likelihood fields of different sizes and resolutions. Also, generate one template (neighborhood of any scan point)
    - For the first scan, we simply add it to the occupancy map, then initialize likelihood fields with it.
2. Iterate through the likelihood field pyramid, from the lowest resolution to the highest:
    1. Build a graph of the pose estimate (vertex) and distance errors of scan points (edges)
    2. Set $\delta$ in $\Chi^2$ of an edge. Above $\delta^2$, a g2o edge is considered an outlier in the data. Then these large errors will be downweighted:
        $$
        \begin{gather*}
        \begin{aligned}
        & \frac{1}{2} (y_i - \hat{y}_i)^2, \text{for} |y_i - \hat{y}_i| < \delta
        \\ &
        \delta |y_i - \hat{y}_i| - \frac{1}{2} \delta^2, \text{otherwise}
        \end{aligned}
        \end{gather*}
        $$
        ```cpp
        auto rk = new g2o::RobustKernelHuber;
        rk->setDelta(delta);
        edge->setRobustKernel(rk);
        ```
    3. Count number of inliers, that is, edges with $Chi^2$ lower than its threshold. If there are not enough inlier, we are not going to add the scan for loop closure / registration
    4. Keep the pose estimate for next level likelihood field optimization
3. Update occupancy grid

Important parameters:

- Matched Scan point (inlier) ratio:
    - If this ratio is too high, a loop is considered not detected and we will lose potential constraints there. 
    - If used in scan registration, one might notice that earlier scans are not added. This will decline further scan additions.

- Thresholds of free / occupied cell values for Visualization:
    - This can create a huge impact of the resulted map (TODO: comparison)

### Loop Closure

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/d4473c0d-02b0-43c2-a7b5-0252831bb874" height="300" alt=""/>
       </figure>
    </p>
</div>

Say submap1 has world pose: T_w_m1, submap 2: T_w_m2. a particular scan is T_w_s. The transform between the two submaps are: T_m1_m2. In a pose graph, this is one "adjacent edge". 

We also want to construct a loop closure edge. That is done by:


$$
\begin{gather*}
\begin{aligned}
& T_{m1, m2} = T_{w, s1} T_{s1, s2} T_{s2, w}
\\ &
e = log(T_{m1, m2})
\end{aligned}
\end{gather*}
$$

The Jacobian of the above is dependent on `x, y, theta` of both submap poses. It's quite complex and we can leave that to g2o's auto differentiation.

## Final Submap  Genenration
