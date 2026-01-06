---
layout: post
title: C++ - [Concurrency 7]
date: 2023-06-08 13:19
subtitle:
comments: true
header-img: img/post-bg-unix-linux.jpg
tags:
  - C++
---
## Thread Policy Issues

[std::execution::par_unseq](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) parallel execution policy on line 152. This could be triggering thread priority issues. Let me check the common test base class:

```cpp
        std::for_each(measurement.lidar_full_cloud_->points.begin(), measurement.lidar_full_cloud_->points.end(),
                      [&](const auto &pt) {
                          ASSERT_GE(pt.time, 0.0)
                              << "lidar point timestamp should be greater than or equal to zero";
                          ASSERT_LT(pt.time, lidar_timestamp)
                              << "lidar point timestamp should be earlier than lidar timestamp";
                      });
    };
```

**:** The test used [std::execution::par_unseq](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) (parallel execution policy) which relies on TBB (Threading Building Blocks). TBB attempted to set thread priorities, but the container doesn't have real-time scheduling permissions (`ulimit -r` = 0).

**Fix:** Removed parallel execution from the timestamp validation loop in test_loosely_coupled_lio.cpp, changing from:

- [ ] what scheduling policies are there? Yes, my impl is TBB. What policy does it want to change to?
