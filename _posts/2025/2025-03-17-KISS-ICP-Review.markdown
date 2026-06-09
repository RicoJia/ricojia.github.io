---
layout: post
title: Robotics - [3D SLAM - 6] KISS ICP Review
date: 2025-03-17 13:19:00
subtitle:
header-img: img/post-bg-o.jpg
tags:
  - Robotics
  - SLAM
comments: true
---
## KISS-ICP: Simple Odometry, Done Carefully

The paper’s core claim is simple: you do **not** need a complicated frontend, feature extraction, normals, surfels, IMU fusion, or learned components to get strong LiDAR odometry.

If you do the basics carefully, plain point-to-point ICP is enough to be competitive:

1. predict motion with a constant-velocity model

2. deskew and downsample the current scan

3. align the scan to a local voxel map using robust point-to-point ICP

4. update the map

5. adapt the correspondence threshold based on recent prediction error

## 1. Predict the next pose with constant velocity

KISS-ICP starts with a very simple motion model.

The code computes the initial guess as:

```python
initial_guess = last_pose * last_delta
```

In other words, it assumes the robot will keep moving the way it was moving in the previous step.

After ICP finishes, the system compares the predicted pose with the estimated pose. That difference tells it how wrong the motion prediction was. KISS-ICP then uses that error to update its internal threshold model.

So the prediction is not just for ICP initialization. It also drives the adaptive behavior later.

## 2. Register scan-to-map, not scan-to-scan

The current scan is aligned to a **local voxel hash map** built from recently registered points.

This is more stable than only matching the current scan against the immediately previous scan. A local map gives ICP more geometric context and makes the registration less sensitive to one noisy or sparse scan.

The map is bounded in size:

- add new points

- keep only a few points per voxel

- remove far-away voxels

So the map stays local, sparse, and fast.

## 3. Use a simple voxel hash map

The map is just a **hashed voxel grid of points**.

For nearest-neighbor lookup, KISS-ICP searches the current voxel and its 26 neighboring voxels. That gives a fast local search without needing a full global nearest-neighbor structure.

When inserting points, the system limits density per voxel and avoids adding near-duplicates. This keeps the map from growing too dense while preserving enough structure for ICP.

The result is a lightweight local map that is good enough for scan-to-map registration.

## 4. Keep ICP simple: point-to-point is enough

KISS-ICP uses robust point-to-point ICP.

No learned features. No surface normals. No surfels. No complicated geometric frontend.

The key is that the system feeds ICP a decent initial guess, uses a stable local map, downsamples the scan, and keeps correspondence search under control.

That combination makes plain ICP work surprisingly well.

## 5. Adapt the correspondence threshold online

Classic ICP needs a max correspondence distance. If the threshold is too large, ICP accepts bad matches. If it is too small, ICP may reject useful matches and fail when motion is harder.

KISS-ICP’s key idea is that this threshold should reflect **how wrong the motion prediction has recently been**.

So it tracks recent prediction error and turns that into an adaptive sigma. ICP then uses roughly:

```python
max_correspondence_distance = 3 * sigma
robust_kernel_scale = sigma
```

In practice, this means the correspondence gate expands when motion is harder and shrinks when motion is calm.

That helps the system generalize across different datasets, sensors, and motion profiles.

The code reflects this directly: after each registration step, it measures the deviation between the predicted pose and the estimated pose, then updates the threshold model from that deviation.

## Summary

KISS-ICP is a minimal LiDAR odometry loop that works because the simple pieces are carefully arranged.

It predicts motion with constant velocity, deskews and downsamples the scan, aligns it to a local voxelized point map using robust point-to-point ICP, and adapts its correspondence threshold from recent prediction error.

The main takeaway is not that ICP is new. It is that a clean scan-to-map ICP pipeline, with a good initial guess and adaptive correspondence gating, can be much stronger than expected.
