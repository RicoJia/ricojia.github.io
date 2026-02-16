---
layout: post
title: Point Cloud Processing
date: 2023-07-21 13:19
subtitle:
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Linux
---

## Point Cloud Filtering with PCL: Extract, Crop, and Clean

When working with point clouds in robotics (e.g., perception pipelines in ROS 2), raw sensor data is almost never used directly. It’s typically too dense, too noisy, and too large. The Point Cloud Library (PCL) provides powerful tools to remove unwanted points, restrict spatial regions, and clean outliers.

## 1️⃣ Removing a Subset of Points with `ExtractIndices`

Sometimes **you already know which points you want to remove**. For example:

- Points belonging to the robot body

- Points classified as ground

- Points flagged by a segmentation step

PCL’s `ExtractIndices` lets you either keep or remove a specific index set.

```cpp
pcl::ExtractIndices<pcl::PointXYZ> extract;  
extract.setInputCloud(filtered_cloud);  
extract.setIndices(to_remove);  // provides the subset.
extract.setNegative(true);        // keep everything NOT indexed  
extract.filter(*filtered_cloud);  // overwrite original cloud
```

Use this when: you’ve already segmented something. You want precise removal.

## 2️⃣ Spatial Filtering with `CropBox`

A `CropBox` filter keeps only points inside a 3D bounding box.

In your example, the boundaries are defined as:

```cpp
Eigen::Vector4f tmp_max(cloud_crop_max_depth_,  
                        2.0 * target_standoff_distance_,  
                        cloud_crop_height_ / 2.0,  
                        0.0);  
  
Eigen::Vector4f tmp_min(cloud_crop_min_depth_ + camera_to_base_link_offset_,  
                        -2.0 * target_standoff_distance_,  
                        -cloud_crop_height_ / 2.0,  
                        0.0);
```

### Why 4D vectors?

PCL uses `Eigen::Vector4f` because internally it operates in homogeneous coordinates. The fourth component (`w`) is typically `0.0` for points.

## 3️⃣ Cleaning Noise with `StatisticalOutlierRemoval`

`StatisticalOutlierRemoval` (SOR) removes points that are statistically inconsistent with their neighborhood.

1. For each point:

    - Find its **k nearest neighbors**

    - Compute the mean distance to those neighbors → did_idi​

2. Compute global statistics:

    - Mean μ of all distances​

    - Standard deviation σ

3. Remove points satisfying:

$$
d > \mu + threshold(\sigma)
$$
