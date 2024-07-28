---
layout: post
title: C++ - Erase Remove
date: '2023-01-06 13:19'
excerpt: An efficient in-place way to remove elements in a container
comments: true
---

## Introduction

Example:

```cpp
void filter_orb_result_with_valid_depths(
    const cv::Mat &depth_img,
    const SLAMParams &slam_params,
    ORBFeatureDetectionResult &res
){
    auto remove_invalid_orb_features = [&](const cv::KeyPoint& keypoint){
        auto p = keypoint.pt;
        double depth = depth_img.at<float>(int(p.y), int(p.x));
        if (std::isnan(depth) || depth < slam_params.min_depth ||
            depth > slam_params.max_depth)
            return true;
        return false;
    };
    auto new_end = std::remove_if(
        res.keypoints.begin(), 
        res.keypoints.end(), 
        remove_invalid_orb_features
    );
    res.keypoints.erase(new_end, res.keypoints.end());
}
```