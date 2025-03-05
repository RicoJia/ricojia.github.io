---
layout: post
title: C++ - Algorithm Functions
date: '2023-01-20 13:19'
subtitle: minmax_element, min_element, reduce, transform
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

### `std::min(v1, v2)`

## `std::transform`

`std::transform` is a funtion that allows us to apply a function (usually lambda) on one or two (zipped) containers, and puts the output into an output container.

1. unary operations

```cpp
// Definition
transform(Iterator inputBegin, Iterator inputEnd, 
        Iterator OutputBegin, unary_operation) 
// example: increment a number in an array
int arr[] = {1,2,3,4}; 
n = sizeof(arr)/sizeof(int); 
std::transform(arr, arr+n, arr, [](int x){return x+1;})
```

2. binary operation

```cpp
// Definition
transform(Iterator inputBegin1, Iterator inputEnd1, Iterator inputBegin2, 
        Iterator outputBegin, binary_operation)
// Example: arr_1 - arr 2 
int arr_1[] = {1,2,3,4}; 
int arr_2[] = {1,2,3,4}; 
int result[5]; 
std::transform(arr_1, arr_1+4, arr_2, result, [](int i1, i2){return i1 - i2; }); 
```

- `std::transform(It1 it1_begin, It1, it1_end, It2 it2_end, It3 out_begin, [](const Type1& a, const Type2&b){return something})`

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

## `std::accumulate`

1. simple summing

```cpp
#include <numeric>
sum = std::accumulate(vec.begin(), vec.end(), 0.0); 
```

- Note: in the below example, sum will be casted to int, even if `std::vector<double> vec`:

```cpp
sum = std::accumulate(vec.begin(), vec.end(), 0); 
```

2. Other accumulative tasks:

```cpp
T accumulate(Iterator first, Iterator Last, T init, Binary_Operation op){
    for(; first != last; ++first){
    init = op(std::move(init), *first); 
    }
}
```

- E.g,

```cpp
std::accumulate(vec.begin(), vec.end(), 1, [](int product, int b){return product*b}); 
```