---
layout: post
title: Robotics - Point Cloud Processing
date: '2024-04-07 13:19'
subtitle: Bruteforce KNN Search, KD Tree
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Overview

Here is a [post about the LiDAR Working Principle and different types of LiDARs](https://ricojia.github.io/2024/10/22/robotics-3D-Lidar-Selection/).

One lidar model is the Range-Azimuth-Elevation model (RAE)

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/f6b078a2-9ff7-49e0-8cab-a841b595075e" height="300" alt=""/>
       </figure>
    </p>
</div>

Its cartesian coordinates are:

$$
\begin{gather*}
\begin{aligned}
& [r cos(E) cos(A), r cos(E) sin(A), r sin(E)]
\end{aligned}
\end{gather*}
$$

If given $[x,y,z]$ coordinates, it's easy to go the other way, too:

$$
\begin{gather*}
\begin{aligned}
& r = \sqrt{x^2 + y^2 + z^2}
\\ &
A = arctan2(\frac{y}{x})
\\ &
E = arctan2(\frac{z}{r})
\end{aligned}
\end{gather*}
$$

Some point clouds have meta data such as reflectance (intensity), RGB, etc. Reflectance can be used for ground detection, lane detection. Based on the above, LiDAR manufacturers transmit **packets** (usually) through UDP after measurements. Velodyne HDL-64S3 is a 64 line LiDAR. At each azimuth angle azimuth, (rotational position in the chart), a message contains on laser block that contains 32 lines (along the elevation angle). Its packet looks like:

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github.com/user-attachments/assets/cc966972-8fc2-4129-accb-358e20f506fd" height="300" alt=""/>
       </figure>
    </p>
</div>

- So, to find the `[x,y,z]` of a single lidar point,
    - `elevation_angle = 32*block_id + array_index`
    - `azimuth = rotational_position * 0.01 deg / 180 deg * pi` (Velodyne lidar azimuth increments in 0.01 deg)
    - `range = array[array_index]`
- In total, we need `2 + 2 + 32 *3 = 100 bytes` to represent the LiDAR Data. If we use the `pcl::PointCloud`, each point would be [x, y, z, intensity], and that's `32 * (4 + 4 + 4 + 1) = 416bytes`. **So packets are compressed LiDAR data.**

Aside from 3D Point Cloud, another 3D representation is a Surface Element Map (Surfel). A surface element is a small 3D patch that contains: 

- `x,y,z`
- Surface normal vector
- Color / texture
- Size / Radius
- Confidence

## Brute Force KNN Search

KNN search is very commonly used in SLAM. A tiny bit of performance improvement can create a huge difference! Therefore, we must consider parallelism in implementation. Brute-Force KNN is very easy to parallelize, and it could be more efficient than more sophisticated algorithms. 

Brute force KNN is:

1. Given a point, search through all points to find their distances. Find the K shortest distances

Of course, we need an extra sorting process to find the k shortest distances

## Pixel & Voxelized Data Structures to Store Point Cloud

Brute force is a good starting point for KNN. To speed up, there are two ways:

1. Point cloud is spatial. We can store them in spatial representations for faster indexing. (This section)
2. We can store a point cloud in a tree, a binary-search tree /KD tree, or a quad-tree or octo tree.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/d90b52d6-32d6-472e-940f-e1a7a3737cc9" height="200" alt=""/>
    </figure>
</p>
</div>

Hash is used to store points into an `unordered_map`