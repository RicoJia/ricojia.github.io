---
layout: post
title: Robotics - [IMU Pre-integration Model 5] IMU Pre-integration Optimization Implementation
date: 2024-04-03 13:19
subtitle: LIO-SAM
comments: true
header-img: img/post-bg-unix-linux.jpg
tags:
  - Robotics
---

## “Tightly coupled LIO” can mean two different things

There are two types of Lio:

- Factor-graph tightly coupled LIO (LIO-SAM), which is easier to extend with backend constraints. You optimize states using factors:
 	- GPS factor
 	- IMU Preintegration factor
 	- lidar odom factor
 	- prior factor
 	- loop closure factor
- Filter-based tightly coupled LIO like FAST-LIO 2:
 	- FAST-LIO fuses LiDAR feature points with IMU data using a tightly coupled iterated EKF. FAST-LIO2 directly registers raw points to the map and is designed for high-rate real-time odometry/mapping. ([arXiv](https://arxiv.org/abs/2010.08196?utm_source=chatgpt.com "FAST-LIO: A Fast, Robust LiDAR-inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter"))
 	- This is not primarily a backend factor graph system. It is more like:
  		- IMU propagation
  		- LiDAR residual update
  		- iterated EKF
  		- local map update
 	- It's great for real-time odometry
 	- loop-closure/global backend is usually a separate layer.

Using IMU pre-integration, instead of optimizing this:

```cpp
pose_i -> imu_1 -> imu_2 -> imu_3 -> ... -> imu_500 -> pose_j
```

you optimize this:

```cpp
state_i -- preintegrated_imu_factor -- state_j
```

where each state usually contains:

- pose
- velocity
- imu_bias

**IMU preintegration factors are easier to plug into existing optimization frameworks** because they turn many high-rate IMU measurements into one compact factor between keyframes. GTSAM’s docs describe preintegrated IMU measurements as summarizing relative motion `ΔR, Δp, Δv` between two time steps and adding that summary as a single `ImuFactor` / `ImuFactor2`, instead of inserting every raw IMU measurement into the graph. ([Borglab](https://borglab.github.io/gtsam/preintegratedimumeasurements/?utm_source=chatgpt.com "PreintegratedImuMeasurements - GTSAM Docs"))

LIO-SAM combines backend + tightly coupled LIO and it is explicitly a **tightly coupled LiDAR-inertial odometry via smoothing and mapping** system. Its implementation has two factor-graph parts:

```cpp
imuPreintegration.cpp:
    IMU preintegration + lidar odometry factor
    estimates pose, velocity, IMU bias

mapOptimization.cpp:
    lidar odometry factor + GPS factor + loop closure
    maintains global map/backend optimization
```

combine backend + tightly coupled LIO

```cpp
front-end scan matching
+ IMU preintegration
+ factor graph backend
+ loop closure / GPS optional
```

---
