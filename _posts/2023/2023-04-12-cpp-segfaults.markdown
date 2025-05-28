---
layout: post
title: C++ - Segfaults
date: '2024-04-12 13:19'
subtitle: Out-of-bounds Iterator Access
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - ROS
---

## Out-of-bounds Iterator Access

Out-of-bounds iterator access is an undefined behavior, which might give a segfault, or give a garbage value. 

```cpp
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v = {10, 20, 30};
    auto iter_first = v.begin();
    auto bad_iter = iter_first + 5;

    // Dereference — UB, likely a segfault
    std::cout << *bad_iter << "\n";
    return 0;
}
```