---
layout: post
title: C++ - Operators
date: '2023-01-10 13:19'
subtitle: Basic Operators
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Basic Operators

- `a == b == c` doesn't do what you want. Do `(a == b && b == c)` instead. It's equivalent to:

```cpp
int a = 5, b = 5, c = 5;
bool res = a == b;
res = res == c;
```
