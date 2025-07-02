---
layout: post
title: Robotics - ROS2 Transform Listener
date: '2024-12-05 13:19'
subtitle: 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Life Cycle

- tf2_ros::TransformListener starts its own subscription `/tf`, `/tf_static`:
    1. `node_->create_subscription<tf2_msgs::msg::TFMessage>( … )`
    2. stores that subscription in the same callback group / executor your node is using, and
