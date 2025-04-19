---
layout: post
title: Robotics - [3D SLAM - 2] NDT Lidar Odometry From Scratch
date: '2025-3-4 13:19'
subtitle: Direct NDT, Incremental NDT Lidar Odometry
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - SLAM
comments: true
---

## Direct Lidar Odometry

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://i.postimg.cc/8cT7R7LN/ndt-sputnik.gif" height="300" alt=""/>
        <figcaption><a href=""> Result on ULHK Dataset </a></figcaption>
    </figure>
</p>
</div>

### Lessons learned

- **Need Voxel filtering at the beginning**
- Only add key frames as target. Otherwise, the point cloud will be too clumped
    - For visualization though, we still add filtered pointcloud
- There are 2 strategies to handle Nearby6 when point cloud is sparse:
    - Add up weighted errors all together. The errors could be large (>10^7 for a point cloud with 20k points)
    - Find the best voxel and use that error in the optimization
        - The error there will be small. 
- PCL 1.12 has a bug in its `spinOnce()` function, and PCL 1.13 is slow in `spinOnce()`. PCL 1.14 is much better
