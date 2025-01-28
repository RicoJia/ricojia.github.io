---
layout: post
title: Robotics - [ROS2 Foundation 3] ROS2 Build System
date: '2024-11-19 13:19'
subtitle: Various Things to Note For Building Dockerized ROS2 App
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
    - Docker
comments: true
---

## Build Tool 

A build tool operates on **a set of packages**
1. determines the dependency graph 
2. invokes the specific build system for each package in topological order.
3. for a specific package, knows how to setup the environment for it, invokes the build, and sets up the environment to use the built package. 

The build system operates on a single package: `CMake`, `Make`, `Python setuptools`. `catkin` and `ament_cmake` are based on `CMake`

### Dependency Graph

- `find_package` helps the graph. `FindFoo.cmake` or `FindFoo.cmake` for the dependency must be in a prefix that CMake searches implicitly, like `/usr`, or a location provided through env vars `CMAKE_PREFIX_PATH`, or `CMAKE_MODULE_PATHCMAKE_MODULE_PATH`
- Install a shared_lib in a non-default location, that location needs to be in `LD_LIBRARY_PATH`.

