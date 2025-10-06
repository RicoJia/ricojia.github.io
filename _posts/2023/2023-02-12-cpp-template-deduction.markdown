---
layout: post
title: C++ - [Templates 3] Template Deduction
date: '2023-02-12 13:19'
subtitle: Template Argument Deduction
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Template Argument Deduction

Direct template argument substitution works! So template argument reduction could work in this case.

```cpp
template <class LidarMsg>
void process_pt_cloud(const std::shared_ptr<LidarMsg>& msg);
sensor_msgs::msg::PointCloud2::SharedPtr pc2_msg;   // std::shared_ptr<sensor_msgs::msg::PointCloud2>
process_pt_cloud(pc2_msg);

// Fine, LidarMsg = sensor_msgs::msg::PointCloud2 ✅
```

A qualified-id is not deducible in templates, because the compiler needs to "reverse-engineer" to check `LidarMsg`

```cpp
template <class LidarMsg>
void process_pt_cloud(const typename LidarMsg::SharedPtr& msg);

using SharedPtr = std::shared_ptr<PointCloud2>;

sensor_msgs::msg::PointCloud2::SharedPtr pc2_msg;
process_pt_cloud(pc2_msg);
```

In this case, `?::SharedPtr = sensor_msgs::msg::PointCloud2::SharedPtr`. The compiler is NOT smart enough to look that up reversely.
