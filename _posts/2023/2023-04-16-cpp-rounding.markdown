---
layout: post
title: C++ - Rounding
date: '2024-04-18 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - ROS
---

## llround

llround follows round half away from zero (not bankers rounding)., **and returns long long**

```cpp
std::llround(2.3)    → 2
std::llround(2.5)    → 3
std::llround(2.7)    → 3

std::llround(-2.3)   → -2
std::llround(-2.5)   → -3
std::llround(-2.7)   → -3
```

std::round returns double

```cpp
std::round(2.5);   // 3.0 (double)
```

- Banker's rounding is to round .5 to the nearest even integer

```
2.5 2 ← (2 is even), 
2.6 → 3
3.5 4 ← (4 is even)
```
