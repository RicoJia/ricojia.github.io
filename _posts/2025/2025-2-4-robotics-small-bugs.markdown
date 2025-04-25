---
layout: post
title: Robotics - [Bugs - 2] SLAM Related Small Bugs
date: '2025-2-1 13:19'
subtitle: Passing Ptr By Value
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - SLAM
comments: true
---

## ⚠️ Beware of Passing Smart Pointers by Value

Passing a smart pointer (e.g. std::shared_ptr or boost::shared_ptr) by value is fine when you're only modifying the object it points to. But if you intend to reassign or reset the pointer itself (e.g. with .reset()), the changes won’t be visible to the caller — you're just modifying a copy. Example:

```cpp
void extract(const PCLFullCloudPtr full_cloud, PCLCloudXYZIPtr edge_points, PCLCloudXYZIPtr planar_points) const{
    if (edge_points == nullptr) {
        edge_points.reset(new PCLCloudXYZI);
    }
    if (planar_points == nullptr) {
        planar_points.reset(new PCLCloudXYZI);
    }
    ...
}
PCLCloudXYZIPtr edge_points = nullptr;
PCLCloudXYZIPtr planar_points = nullptr;
extractor.extract(scan_ptr, edge_points, planar_points);`
```