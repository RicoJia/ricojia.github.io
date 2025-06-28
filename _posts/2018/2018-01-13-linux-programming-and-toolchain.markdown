---
layout: post
title: Linux - Programming And Toolchain
date: '2018-01-13 13:19'
subtitle: ldd
comments: true
tags:
    - Linux
---

## ELF and Dynamic Linking

### ldd - Inspect Shared-Library Dependencies

`ldd some_executable` asks the dynamic linker to resolve each shared object name, and prints `libsomething.so.1 => /actual/path/libsomething.so.1 (0xaddress)`

```bash
└─  $ ldd ./lib/libbehavior_executor.so
 linux-vdso.so.1 (0x0000728663d27000)
 libdiagnostic_msgs__rosidl_typesupport_cpp.so => /opt/ros/humble/lib/libdiagnostic_msgs__rosidl_typesupport_cpp.so (0x0000728663aaa000)
```
