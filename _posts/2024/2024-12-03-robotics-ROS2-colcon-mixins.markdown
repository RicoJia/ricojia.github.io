---
layout: post
title: Robotics - ROS2 Colcon Mixins
date: '2024-11-30 13:19'
subtitle: Group colcon build arguments under a single name
header-img: "img/post-bg-os-metro.jpg"
tags:
    - Robotics
    - ROS2
comments: true
---

## What is a Mixin?

A _mixin_ is **a reusable piece of functionality that you can "mix into" something else**.

In Python, a class mixin adds behavior to a class:

```python
class LoggerMixin:
    def log(self, msg):
        print(msg)

class MyClass(LoggerMixin):
    pass
```

Similarly, a **colcon mixin** adds a named group of arguments to a colcon command:

- Class mixin → adds behavior to code
- Colcon mixin → adds options to a command

[Colcon Mixins](https://github.com/colcon/colcon-mixin-repository) allow you to apply a group of arguments under one name. For example:

```bash
colcon build --mixin debug clang
```

This might expand to flags such as:

```
--cmake-args -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
```

> **Note:** Mixins are applied in the order listed. If two mixins set the same option, the later one overrides the earlier one.

## Installation

```bash
sudo apt install python3-colcon-mixin
```

## How To Use Mixins

### 1. Define YAML Files

File structure:

```
my_mixins/
├── index.yaml
├── debug.yaml
└── clang.yaml
```

`index.yaml` — maps mixin names to their files:

```yaml
build:
  debug: debug.yaml
  clang: clang.yaml
```

`debug.yaml`:

```yaml
arguments:
    cmake-args:
        - -DCMAKE_BUILD_TYPE=Debug
```

`clang.yaml`:

```yaml
arguments:
    cmake-args:
        - -DCMAKE_C_COMPILER=clang
        - -DCMAKE_CXX_COMPILER=clang++
```

### 2. Register the Mixins

```bash
colcon mixin add my_mixins /path/to/my_mixins/index.yaml
colcon mixin update
```

### 3. Verify

```bash
colcon mixin show
```

### 4. Use

```bash
colcon build --mixin debug clang
```

## Useful Commands

Remove build artifacts for specific packages:

```bash
# Remove a specific package
colcon clean packages -y --packages-select <package_name>

# Remove a package and all packages that depend on it
colcon clean packages -y --packages-up-to <package_name>
```

The `--packages-up-to` flag also works with mixins.
