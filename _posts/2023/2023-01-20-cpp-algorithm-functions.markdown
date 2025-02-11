---
layout: post
title: C++ - Algorithm Functions
date: '2023-01-20 13:19'
subtitle: minmax_element
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## `std::minmax_element(begin_itr, end_itr, lambda)`

This function finds the `[min, max]` of a container based on the definition of "smaller".

```cpp
#include <algorithm>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};

    auto minmax = std::minmax_element(vec.begin(), vec.end(), [](const int& i1, const int& i2){
        return i1 < i2;
        // return i1>i2; // the first element (min) is the one that satisfies this condition throughout the container
    });

    // See Min element: 1
    std::cout << "Min element: " << *minmax.first << std::endl;
    std::cout << "Max element: " << *minmax.second << std::endl;

    return 0;
}
```