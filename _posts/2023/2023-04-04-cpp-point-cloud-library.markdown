---
layout: post
title: Point Cloud Library
date: '2024-04-04 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - ROS
---

## Point Infrastructure

### Point Cloud Members

[`PointCloud` Type includes](https://pointclouds.org/documentation/singletonpcl_1_1_point_cloud.html)

- `points`: point vector

### PointTypes

`PointType` include:
- pcl::PointXYZ
- pcl::PointXYZI
- pcl::PointXYZRGB
- pcl::PointXYZRGBA
- pcl::PointNormal
- pcl::PointXYZRGBNormal

[Below convenience functions can be found on this page.](http://docs.ros.org/en/hydro/api/pcl/html/point__types_8hpp.html)

- `res=PointType.getVector3fMap()`
    - Since this returns an `Eigen::vector3f`, one can use `res.squaredNorm()`further.

### Point Cloud Width and Height

- `width` – how many points make up one row of the cloud
- `height` – how many rows there are
- `points.size()` – the total number of points = `width * height`