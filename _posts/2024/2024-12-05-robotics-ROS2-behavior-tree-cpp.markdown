---
layout: post
title: Robotics - ROS2 Behavior Tree Cpp
date: '2024-12-05 13:19'
subtitle: 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---


## Nodes

- A `CoroAction` is to pause, send an action goal, wait, get an action result, resume.
- destructor order:
    1. the dtor body
    2.members in a reverse order
