---
layout: post
title: Robotics - ROS2 Tools
date: '2024-11-5 13:19'
subtitle: Ros2 Doctor, Topic, Ros2 Multicast, package finder
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## ROS 2 Doctor

[References: ROS2 Doctro](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Getting-Started-With-Ros2doctor.html)

`ROS2 Doctor` is a diagnostic tool that succeeds `ros wtf` (where is the fault). It checks all aspects of ROS 2, including platform, version, network, environment, running systems. It's part of the `ros2cli` package. So if `ros2cli` is installed, `ROS2doctor` is installed.

### What Does ROS2 Doctor Check?

One can see `ros2 doctor --report`

- Package versioning, such as `tf2_bullet`
- `NETWORK CONFIGURATION`, items like:
  - `inet         : 192.168.1.65`...
- `PLATFORM INFORMATION`, items like:
  - like `platform info    : Linux-6.9.3-76060903-generic-x86_64-with-glibc2.35`
- `QOS COMPATIBILITY LIST`
  - `compatibility status    : No publisher/subscriber pairs found`
- `ROS 2 INFORMATION`:
  - `distribution name      : humble`
- `TOPIC LIST`
  - Existence of "dangling" topics without subscribers.

One can check failed checks only (`report fail`): `ros2 doctor -rf`. [Reference](https://github.com/ros2/ros2cli/tree/foxy/ros2doctor)

### What `ros2 doctor` Can't Detect

- I had a mix of `ROS2 Iron` and `ROS2 Humble`
  - ROS 2 Iron cannot directly listen to ROS 2 Humble messages due to ABI incompatibility and other changes between ROS 2 distributions. E.g.,
    - `sensor_msgs` that has different symbols or behavior in `Iron`

## Ros2 Topic

In ROS 2 design, it's generally good to keep in mind what `QoS` we might need for each topic

- `ros2 topic echo /your_topic --qos-reliability=best_effort`
  - Messages might be dropped if the subscriber cannot keep up or if the network connection is unreliable.
- Can use `ros2 node info /wayfinder/head_stereo_camera/depth_left_camera` to see which nodes are publishing / subscribing via this topic
- Checking pub-sub **qos**: `ros2 topic info TOPIC --verbose`

- `ros2 topic hz` currentlty doesn't have a `qos` option
- Useful: `ros2 topic echo /diagnostics`: monitors if there are dropped frames, overheating, etc. Any node can publish to `/diagnostics` . (Hardware drivers, software nodes, etc.)

### Multicast

This command is a diagnostic tool that's often used in ROS 2 for discovery and communication between nodes

- `ros2 multicast receive`

## ROS2 pkg

- Find the path to a package: `ros2 pkg prefix PKG`
