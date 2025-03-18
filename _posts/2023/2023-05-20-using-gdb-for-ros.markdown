---
layout: post
title: ROS - GDB
date: '2023-05-20 13:19'
subtitle: GDB for ROS, Core Dump
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - ROS
---

## Introduction

GDB is **EXTREMELY** handy. I used to doing a lot of print statements, but now I use GDB a lot. The reason is simple: I can check multiple things at once without recompiling everytime, especially my compilation time is long.

## Concepts

Debug Symbols in C or C++ are mappings from machine code to function names, variable names, line numbers, etc.

E.g., in a binary, `i` is just a memory location. When built with a `Debug` flag (see the Usage section), we know the variable's name.

```cpp
int i = 1;
```

Debugging symbols do not impact the binary execution logic. It's purely informational and consumed by gdb. The binary filesize though would increase.

### Compilation 

- In `CMakeLists.txt`

```cmake
set(CMAKE_BUILD_TYPE Debug)
```
    - Or alternatively, `catkin_make -DCMAKE_BUILD_TYPE=Debug`

- In `gcc`:

```bash
gcc -g -o myprog myprogram.c
```

### Run GDB

- In a ROS Launch file:

```xml
<node pkg="rgbd_slam_rico" type="rgbd_rico_slam_mini" name="rgbd_rico_slam_mini" output="screen" launch-prefix="gdb -ex run --args"/>
```

- Alternatively, you can run the binary using the command `gdb`:

```bash
gdb ./devel/lib/<PACKAGE_NAME>/binary
```

## GDB Usage

Some common usages include:

- `b <FILE_NAME>:<LINE_NUM>` set a breakpoint
- `r`: run 
- `c`: continue
- `p <var_name>`: print a variable name
- `info breakpoints`: check which breakpoints are available
- `delete <BREAKPOINT_NUM>`: to delete a break point's number

Some cautions:

- When debug printing an element in a `cv::Mat`, we need to specify its data type. To print it as a: 
    - double (`CV_64F`), do `(gdb) x/4g var`. 4 means the first 4 values, `g` means double precision. 
    - float (`CV_32F`): `(gdb) x/4f var`
    - int: `(gdb) x/4d var`
