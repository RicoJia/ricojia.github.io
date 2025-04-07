---
layout: post
title: Robotics - [3D SLAM - 1] Lidar Odometry
date: '2025-3-4 13:19'
subtitle: Direct Lidar Odometry, Incremental Lidar Odometry
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - SLAM
comments: true
---

## Direct Lidar Odometry

When given two scans, a direct NDT method means no features are extracted. The two scans would directly go through NDT for finding their relative poses. The workflow is as follows

```
scan -> NDT ---pose_estimate---> Is_Keyframe? ---yes---> Add_Frame
```

### NDT Odometry

1. NDT:
    1. *Build hash map that represents voxels from all point clouds of the current submap* [**INEFFICIENT**]
    2. Calculate the mean and variance of the voxels
    3. Update motion_model = `last_pose.inverse() * pose`
        - This can be used as the pose initialization

2. Is_Keyframe:
    1. If the keyframe has travelled over an angular / distance threshold
    2. If the keyframe has travelled over an angular / distance threshold

3. Add_Frame
    1. We cscan -> NDT -> Is_Keyframe? ---yes---> Add_Frame

## Incrememental NDT Odometry

The high level workflow of Incremental NDT is the same as the non-incremental one:

```
scan -> NDT ---pose_estimate---> Is_Keyframe? ---yes---> Add_Frame
```

The main difference, however, is in `add_frame`:
1. Given two point clouds: source and target, we can voxelize them. 
2. For the same voxel location in source and target:
    1. Count the number of points in the voxel of source and target: `m`, `n`
    2. We can calculate mean and variances of them: $\mu_a$, $\mu_b$, $\Sigma_a$,$\Sigma_b$
    3. Now we want to add the source cloud to the target. We can update the target point cloud's new mean directly:

        $$
        \begin{gather*}
        \begin{aligned}
        & \mu = \frac{x_1 + ...x_m + y_1 + ... y_m}{m + n} = \frac{m \mu_a + n \mu_b}{m + n}
        \end{aligned}
        \end{gather*}
        $$

    4. For the target's new variance (ignoring Bessel correction):
        $$
        \begin{gather*}
        \begin{aligned}
        & \Sigma = \frac{1}{m+n} (\sum_i (x_i - \mu)(x_i - \mu)^T + (y_i - \mu)(y_i - \mu)^T)
        \\ &
        \sum_i (x_i - \mu)(x_i - \mu)^T = \sum_i (x_i - \mu_a + (\mu_a - \mu))(x_i - \mu_a + (\mu_a - \mu))^T
        \\ & = \sum_i [(x_i - \mu)(x_i - \mu)^T + (x_i - \mu_a)(\mu_a - \mu)^T + (\mu_a - \mu)(x_i - \mu_a)^T + (\mu_a - \mu)(\mu_a - \mu)^T]
        \\ & \text{one can see that}
        \sum_i (x_i - \mu_a)(\mu_a - \mu)^T = 0 = \sum_i (\mu_a - \mu)(x_i - \mu_a)^T
        \\ & \Rightarrow
        \sum_i (x_i - \mu)(x_i - \mu)^T = m \Sigma_a

        \\ & \text{So ultimately:}
        \\ & \Sigma = \frac{1}{m+n}[m (\Sigma_a + (\mu_a - \mu)(x_i - \mu_a)^T) + n(\Sigma_b + (\mu_b - \mu)(x_i - \mu_b)^T)]
        \end{aligned}
        \end{gather*}
        $$

3. Also, if we want to discard voxels if there have been too many old ones, we can implement an Least-Recently-Used (LRU) cache of voxels. 

- If there are too many points in a specific voxels, we can choose not to update it anymore.

## Indirect Lidar Odometry

Indirect Lidar Odomety is to select "feature points" that can be used for matching. There are 2 types: **planar** and **edge feature** points. This was inspired by LOAM (lidar-odometry-and-mapping), and is adopted by subsequent versions (LeGO-LOAM, ALOAM, FLOAM). Indirect Lidar Odometry is the foundation of LIO as well. Common features include: PFH, FPFH, machine-learned-features. Point Cloud Features can be used for database indexing, comparison, compression.

After feature detection, we can do scan-matching using ICP or NDT

Some characteristics of 3D Lidar points are:

- They are not as dense as the RGB-D point cloud. Instead they have clear **line characteristics**. 

Questions:

- 







**There you have it, NDT odometry. Note that due to the lack of loop detection, it's still susceptible to accumulated error.**