---
layout: post
title: RGBD SLAM - Building A Docker Container For ROS2 And Object Detection Deep Neural Nets
date: '2024-08-01 13:19'
subtitle: A Summary Of Setting Up A ROS 2 Docker Container For Object Detection Training And Inferencing
header-img: "img/post-bg-unix"
tags:
    - RGBD Slam
    - ROS2
    - Deep Learning
comments: true
---

### Docker Runtime Args

- `--runtime=nvidia`: enable Nvidia Container Runtime, a "runtime" that connects Nvidia GPU with docker. If your laptop doesn't have an Nvidia GPU, simply remove this arg.