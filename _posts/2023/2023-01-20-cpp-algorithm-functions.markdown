---
layout: post
title: C++ - Algorithm Functions
date: '2023-01-20 13:19'
subtitle: minmax_element, min_element
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

### `std::min_element`: returns the min element based on the defintion of "smaller"

## `std::reduce` (C++17)

`std::reduce` is an algorithm introduced in C++17 (and enhanced in C++20) that aggregates a range of elements using a binary operation and an initial value. 

**Unlike `std::accumulate`, `std::reduce` has no guaranteed order of evaluation, which allows it to be parallelized for improved performance.** However, because it processes elements in an unspecified order, the binary operation should be **both associative and commutative** to guarantee correct results.

```cpp
#include <optional>
#include <iostream>
#include <vector>
#include <numeric>
using namespace std;

void test_reduce(){
    std::vector<int> vec {1,2,3,4,5};
    // order of execution is unspecified
    double sum = std::reduce(
        std::execution::par_unseq,
        vec.begin(), vec.end(), 0.0, [](int a, double sum){return 0.25*a + sum;});
    cout<<"using reduce to find sum: "<<sum<<endl;
}

int main(){
    // test_optional();
    test_reduce();

}
```

- `std::execution::par_unseq` parallelizes the above.
-  Pay attention to the initial value `0.0`, otherwise, it will be an int and will be rounded.