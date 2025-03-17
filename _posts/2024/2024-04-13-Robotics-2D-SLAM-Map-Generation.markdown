---
layout: post
title: Robotics - [2D SLAM 2] Map Generation
date: '2024-04-13 13:19'
subtitle: Submap Generation
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
    - SLAM
---

## Submap Generation

In modern robotics mapping systems, efficient environment representation and robust localization are achieved by dividing the map into **smaller, manageable submaps**. Each submap is built upon two key components: an occupancy map that records obstacles and free space, and a likelihood field derived from the occupancy map. The workflow of submap generation is as follows:

1. Initialize the occupancy map with the first scan. 
2. Generate a likelihood field according to the scan
3. For subsequent scans:
    1. Perform Scan-match against the likelihood to find the pose estimate from the last scan `T_21`. 
    2. If `T_21` shows a translation or a rotation larger than their thresholds. The current scan is a **keyframe**. 
    3. Add the keyframe into the occupancy map. 
    4. Generate a likelihood field according to the scan
    5. Create a new submap with its origin being the current scan's origin, if any below condition is met:
        1. The occupancy map already has `N` submaps
        2. Any point already falls outside of the map boundary. 

## Loop Closure

## Final Submap  Genenration
