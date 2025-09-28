---
layout: post
title: Robotics - [3D SLAM - 5] Loosely and Tightly Coupled Lidar Inertial Odometry
date: '2025-3-11 13:19'
subtitle: FastLIO
header-img: "img/post-bg-o.jpg"
tags:
    - Robotics
    - SLAM
comments: true
---

# Loosely vs Tightly Coupled LIO

In loosely coupled LIO, the IMU and GPS are first fed into error state kalman filter which outputs a global pose estimate `T_G`. `T_G` is fed into a lidar odometer (such as `LOAM`) as an initial relative pose estimate. The lidar odometer will iterative optimize the pose estimate to minimize its residual, and outputs a refined pose `T_R`. `T_R` is fed back into the ESKF as an SE3 estimate (just like GPS).

In tightly coupled lio, The Error state kalman filter has an embedded lidar odometer with iterative pose optimization. IMU, GPS, and lidar residuals are added to the ESKF as individual observations.

Therefore, loosely coupled LIO keeps the ESKF and lidar odometer relatively separate, while tightly coupled LIO merges the two together.  

## Tightly Coupled Lio

### FastLIO

Highlights:

- It terms "integration model" as forward prop
- Motion undistort is just the intuitive model that aligns all points to the same timestamp.
