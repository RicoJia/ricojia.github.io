---
layout: post
title: Robotics - ROS2 Colcon Mixins
date: '2024-11-30 13:19'
subtitle: 
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## Introduction

[Colcon Mixins](https://github.com/colcon/colcon-mixin-repository) can be used to apply a group of arguments under one name. Example:

```
colcon build --mixin debug clang
```

This might apply a group of flags related to debug builds, such as:

```
--cmake-args -DCMAKE_BUILD_TYPE=Debug
```

- Note: mixins are applied in the order listed. If two mixins set the same option, the later one can override the earlier one.

**To download**:

```
sudo apt install python3-colcon-mixin
```

## How To Use Mixins

1. Define yaml file for mixins:

    - File structure:

        ```
        my_mixins/
        ├── index.yaml
        ├── debug.yaml
        └── clang.yaml
        ```

    - `index.yaml`

        ```yaml
        build:
        debug: debug.yaml
        clang: clang.yaml
        ```

    - `debug.yaml`:

        ```yaml
        arguments:
            cmake-args:
                - -DCMAKE_BUILD_TYPE=Debug
        ```

    - `clang.yaml`

        ```yaml
        arguments:
            cmake-args:
                - -DCMAKE_C_COMPILER=clang
                - -DCMAKE_CXX_COMPILER=clang++
        ```

2. Register the mixins:

```bash
colcon mixin add my_mixins /path/to/my_mixins/index.yaml
colcon mixin update
```

3. To check:

```bash
colcon mixin show
```

4. Use mixins:

```bash
colcon build --mixin debug clang
```
