---
layout: post
title: C++ - [Container 2] Vector Operations
date: '2023-01-29 13:19'
subtitle: `std::iota`, `std::generate`
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Filling Operations

- Fill a vector with `0, 1,2,3,...`: `std::iota`. `iota` is the greek word for "small amount"

```cpp
#include <numeric>
std::vector<int> vec(10);
std::iota(vec.begin(), vec.end(), 0);
```

- Fill a vector with values with a custom lambda:

```cpp
#include <algorithm>
std::vector<int> vec(10,2); // now all 2
int x = 1;
std::generate(vec.begin(), vec.end(), [&x](){return x*=3;}); // 3, 9, ...
```
    - for `std::generate()`, the lambda must be a callable without args

- Fill the first `n` element with values:

```cpp
#include <algorithm>
std::vector<int> vec(10,2); // now all 2
int x = 1;
int n = 5;
std::generate_n(vec.begin(), n, [&x](){return x*=3;}); // 3, 9, ...
```