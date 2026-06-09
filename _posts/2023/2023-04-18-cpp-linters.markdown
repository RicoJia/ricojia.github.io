---
layout: post
title: C++ - Specifiers
date: 2024-04-18 13:19
subtitle: Ament CPP
comments: true
header-img: img/post-bg-alitrip.jpg
tags:
  - ROS
---
## What Ament Cpp Actually Does

`ament_cppcheck` is a ROS 2 wrapper around Cppcheck for static analysis of C/C++ source files, and ROS documents it as a lint/static-analysis tool, not as something that changes or validates the optimized binary itself. ROS also notes that some Cppcheck versions can be slow, which is one reason it may be disabled on some systems.

In ROS 2 packages, `ament_lint_auto` / `ament_cppcheck` are normally wired under `if(BUILD_TESTING)` and declared as `test_depend`, so they behave like test/lint dependencies rather than runtime or release-build dependencies. The `ament_lint_auto` docs show this pattern directly with `if(BUILD_TESTING) ... ament_lint_auto_find_test_dependencies()`

The actual analysis comes from **Cppcheck**, which focuses on bugs, undefined behavior, and dangerous coding constructs in C/C++ code. Examples include:

- possible null pointer dereferences
- uninitialized variables
- out-of-bounds / buffer overrun issues
- memory leaks
- mismatched allocation/deallocation
- use-after-free / invalid lifetime patterns where detectable
- dead code or unreachable/redundant conditions
- suspicious conditionals, assignments, or comparisons
- some portability / undefined-behavior issues
