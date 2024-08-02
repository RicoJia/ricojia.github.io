---
layout: post
title: C++ - Using GDB for ROS
date: '2023-05-20 13:19'
excerpt: GDB is a very useful that allows us to pause at a break point or failure and inspect.
comments: true
---

## Concepts

Debug Symbols TODO

## Usage

- In `CMakeLists.txt`

```cmake
set(CMAKE_BUILD_TYPE Debug)
```
    - Or alternatively, `catkin_make -DCMAKE_BUILD_TYPE=Debug`

- In a ROS Launch file:

```xml
<node pkg="rgbd_slam_rico" type="rgbd_rico_slam_mini" name="rgbd_rico_slam_mini" output="screen" launch-prefix="gdb -ex run --args"/>
```
