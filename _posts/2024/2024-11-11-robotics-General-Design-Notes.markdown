---
layout: post
title: Robotics   General Design Notes 
date: '2024-11-11 13:19'
subtitle: What's New In ROS2
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Software Structuring

- Build a simulation / dataset for debugging software is key. otherwise, you will have a lot of overhead of on-robot hardware testing.
- Have three general feature flags / launch flags of your robot software: 
    - production
    - simulation: no redundant visualization, so you can run simulation with small amount of compute
    - debug: bring up necessary visualization, log messages. the real-time performance could be sub-par, or certain parts could crash due to compute constrants.
