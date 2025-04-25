---
layout: post
title: Robotics - [3D SLAM - 2] Lidar Odometry
date: '2025-3-4 13:19'
subtitle: Direct Lidar Odometry, NDT, Incremental NDT, Indirect Lidar Odometry
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

I'm implementing a robust 3D NDT-based Odometry system, and it's been a rewarding challenge! Here's a quick breakdown of how it works, plus some lessons learned and visual results.

1. NDT Registration:
    1. *Build hash map that represents voxels from all point clouds of the current submap* [**INEFFICIENT**]
    2. Calculate the mean and variance of the voxels
    3. Update motion_model = `last_pose.inverse() * pose`
        - This can be used as the pose initialization

2. Keyframe Detection:
    1. 📐 Trigger a new keyframe if the pose has shifted significantly in angle or translation

3. Frame Integration Pipeline
    `Scan → NDT Alignment → Keyframe Check → (If Yes) Add to Map`

Here, you can [check out my implementation](https://github.com/RicoJia/Mumble-Robot/blob/main/mumble_onboard/halo/include/halo/lo3d/direct_ndt_3d_lo.hpp)

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/8cT7R7LN/ndt-sputnik.gif" height="300" alt=""/>
        <figcaption><a href=""> My NDT LO Result on ULHK Dataset </a></figcaption>
    </figure>
</p>
</div>

### 💡 Key Takeaways

- ✅ Always voxel-filter input scans in pre-processing — it massively reduces noise and speeds things up.
- 📌 Only keyframes are used to build the target map. Non-keyframes still contribute to visualization but don’t clutter the optimization.
- 🧩 For sparse data, you have two options for Nearby6 voxel association:
    - 🔺 Aggregate all voxel errors (can explode to 10⁷+ for large clouds!)
    - ✅ Use only the best voxel — leads to more stable optimization
- 🐛 PCL versions matter:
    - PCL 1.12 has a `spinOnce()` bug. PCL 1.13 is stable but slow
    - PCL 1.14 is much better

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
    \\ & \Sigma = \frac{1}{m+n}[m (\Sigma_a + (\mu_a - \mu)(\mu_a - \mu)^T) + n(\Sigma_b + (\mu_b - \mu)(\mu_b - \mu)^T)]
    \end{aligned}
    \end{gather*}
    $$

3. Also, if we want to discard voxels if there have been too many old ones, we can implement an Least-Recently-Used (LRU) cache of voxels. 

- If there are too many points in a specific voxels, we can choose not to update it anymore.



### Summary of Vanilla and Incremental NDT Odometry

Similarities:

- We don't store points into point cloud permanently. We just store voxels and their stats

How do we update voxels? 
- NDT3D needs a grid_: `unordered_map<key, idx>`. This means:
    1. We need to reconstruct the grid cells from scratch.
    2. We need keyframes which allows us to selectively add ponitclouds
- NDT3D_inc needs an LRU cache: `std::list<data>`, `unordered_map<itr>`. We don't need to generate grid cell 

## Indirect Lidar Odometry

Indirect Lidar Odomety is to select "feature points" that can be used for matching. There are 2 types: **planar** and **edge feature** points. This was inspired by LOAM (lidar-odometry-and-mapping), and is adopted by subsequent versions (LeGO-LOAM, ALOAM, FLOAM). Indirect Lidar Odometry is the foundation of LIO as well. Common features include: PFH, FPFH, machine-learned-features. Point Cloud Features can be used for database indexing, comparison, compression.

LeGO-LOAM uses distance image to extract ground plane, edge and planar points; mulls uses PCA to extract plane, vertical, cylindircal, horizontal features. 

After feature detection, we can do scan-matching using ICP or NDT

Some characteristics of 3D Lidar points are:

- They are not as dense as the RGB-D point cloud. Instead they have clear **line characteristics**.


- Should be lightweight. Not much CPU/GPU is used
    - So people in industry LO systems use simple features instead of complex ones, learned by machine learning systems.
- Computation should not be split - some on CPU, some on GPU.

Edge points along their vertical direction can be used for point-line ICP; planar points can be used for point-plaine ICP. 

![Image](https://github.com/user-attachments/assets/daafcfa5-7ad2-4beb-acdd-f896a48840ac) TODO

### LOAM-Like Feature Extraction

This line of thoughts can be applied in 2D lidars as well. 

```python
extract(pc_in, pc_out_edge, pc_out_surf):
    scan_in_each_line = [[]]
    for point in pc_in:
        populate scan_in_each_line with points on each line

    for line in scan_in_each_line:
        cloud_curvature = []
        for point in line:
            diff_x = sum(10 neighbors.x) - 10 * point.x
            diff_y = sum(10 neighbors.x) - 10 * point.y
            diff_z = sum(10 neighbors.x) - 10 * point.z
            edgeness = (diff_x^2 + diff_y^2 + diff_z^2)
            cloud_curvature.append(point.id, edgeness)
        # segment 360 deg into 6 areas:
        for segment in 360 deg:
            cloud_curvature_seg = [cloud_curvature[segment.start], ... cloud_curvature[segment.end]]
            # sort
            sort(cloud_curvature_seg, pt.edgeness, DESCENDING_ORDER)
            point_num = 0
            ignored_points = []
            for corner_candidate_point in cloud_curvature_seg:
                if corner_candidate_point in ignored_points:
                    continue
                if corner_candidate_point.edgeness < CORNER_THRESHOLD:
                    break
                if point_num >= NUM_THRESHOLD:
                    break
                pc_out_edge.append(corner_candidate_point)

                for n in left_5_neighbor(corner_candidate_point):
                    if n.x^2 + n.y^2 + n.z^2 > ignore_threshold:
                        # My understanding: We want to ignore flat point around the edge feature, 
                        # this point has too much edgeness
                        # we want to break here, so this point will be added
                        # in the upcoming iteration
                        break
                    ignored_points.append(n)
            for planar_pt_candidate in cloud_curvature_seg:
                if planar_pt_candidate not in ignored_points:
                    pc_out_surf.append(n)
```

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/9Q1Mz9NZ/edge-points.png" height="300" alt=""/>
        <img src="https://i.postimg.cc/YS9SFSJV/feature-points.png" height="300" alt=""/>
        <figcaption> Left: Planar Points, Right: Edge Points</figcaption>
    </figure>
</p>
</div>

Why segment the scan?

- We evaluate a point’s “ruggedness” (local curvature) against its neighbors to choose edge and planar features. Without segmentation, high-curvature areas would dominate, leaving other directions underrepresented. By dividing each scan line into fixed angular sectors (e.g., six), we ensure a balanced quota of features from every part of the scan, yielding uniform spatial coverage.

- Handling degeneracy along a wall
When the sensor travels parallel to a long, flat surface, all strong edges lie in roughly the same direction—this can make the Hessian in our optimizer rank-deficient. Segmentation forces us to pick features from multiple sectors (including those opposite the wall), which restores geometric diversity and prevents loss of observability.

#### Differences From The LOAM Paper

In the original paper:

- First, points are time-warped to the beginning of the scan using either a linear “twist” model or IMU interpolation before curvature is computed (ScanRegistration §3.1)

- For each scan line/beam, divide the point cloud into six sectors (each takes 1/6 of start-end index diff), and extract sharp (corner) and flat (plane/surface) features as follows.

    - In each sector, sort cloud index according to curvature. The top 2 largest curvature points (if points are not selected and curvature > 0.1) are marked as **sharp**, the top 20 largest curvature points are marked as **less sharp** (including the top 2 sharp points), the top 4 smallest curvature points (with additional condition curvature < 0.1) are **marked as flat**, and all the rest points plus the **top 4 flat points** will be down sampled and marked as less flat.
    - After each feature extraction, there will be a **non-maximum suppression** step to mark neighbor 10 points (+-5) to be already selected (marking will stop at points that are 0.05 squared distance away from currently picked point), so that they will not be picked in the next iteration for feature extraction.
    - A downsample operation (of leaf size 0.2m) is applied to each scan of less flat points, and then all downsampled scans are combined together (into one point cloud) to be published to odometry.

- For example for VLP-16 LiDAR, there could be 384 flat feature points and 192 sharp feature points extracted in a point cloud frame (if all meet the 0.1 curvature threshold).

- KD Tree is used for fast indexing

However, the original paper has some drawbacks too:

- Most numbers are “magic constants” set in launch file or hard coded






**There you have it, NDT odometry. Note that due to the lack of loop detection, it's still susceptible to accumulated error.**