---
layout: post
title: C++ - Filesystem
date: '2024-04-14 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - ROS
---

## `std::filesystem::create_directory(path)`

Thread safety:

- `std::filesystem::create_directory(path);` under the hood it issues a single POSIX mkdir() (or the platform equivalent). 
    - If the directory doesn’t exist, one thread’s call will succeed and return true; 
    - Any concurrent or subsequent calls will see that the directory already exists and return false

- Below is a **TOCTOU (Time Of Check To Time Of Use)** race:

```cpp
if (!std::filesystem::exists(path)) {
    std::filesystem::create_directory(path);
}
```
