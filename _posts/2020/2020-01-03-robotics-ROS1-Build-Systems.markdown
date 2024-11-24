---
layout: post
title: Robotics - ROS1 Build Systems
date: '2024-01-03 13:19'
subtitle: catkin build, catkin make
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS1
comments: true
---

## `catkin make`

Catkin make is basically a wrapper of the `CMake` and `make`. It's part of the original Catkin build system, and build the entire workspace using a single `Cmake` invocation.

- `catkin make` builds pacakges all together, so there's less isolation
- Supports parallelism as well, but less fine-grained control over that.
  - `catkin_make -j4` launches 4 sub-processes. Relies entirely on `cmake` parallelism
- Might handle "hidden" dependencies implicitly, whereas `catkin build` requires more explicit dependency declarations.

## `catkin build`

`catkin build` is part of the `catkin_tools` package.

- It provides more parallelism, and builds packages separately (which provides more isolation)
  - Can support parallelism on the number of both build subprocesses and the number of targets. `catkin build -j4 -p2` means "build 2 packages at a time"

- supports build configurations like `release` vs `debug`
- only rebuilds packages that have changed, or dependencies that have changed.

### Useful Commands

- `catkin build --verbose`: Verbose Output
- `catkin clean`: delete `build`, and `devel`, `logs`. This is part of the `catkin tools` package so it won't work for `catkin make`

## Differences

- `${CMAKE_SOURCE_DIR}`
  - `catkin_make` treats this as workspace root
  - `catkin build` finds the package root

## General Guidelines

- Avoid putting non-catkin packages under the `src` directory
  - Adding a `package.xml` to it will break the SLA to keep the package a non-catkin one, so this is not a good idea
  - A good idea in this case is to install it.
    - If you are using docker without root priviledges, you can't install it in global space
    - You can install it in a custom space instead.

```cmake

```
