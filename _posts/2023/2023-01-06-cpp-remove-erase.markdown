---
layout: post
title: C++ - Erase Remove
date: '2023-01-06 13:19'
subtitle: An efficient in-place way to remove elements in a container
comments: true
tags:
    - C++
---

## Introduction

It's a common task to remove items in a sequential container that fits a criteria, like an array, vector, queue, etc. In C++, it's advised to use the **erase-remove** idiom, which does **in-place** element swap and remove.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>    // for std::remove_if

using namespace std;    // don't be like me in production code, I'm being lazy :)

int main()
{
    std::vector<int> vec {1,2,3,5,4,0};
    auto start = std::chrono::high_resolution_clock::now()
    auto valid_end = std::remove_if(vec.begin(), vec.end(), [](const int& i){return i > 3;});
    vec.erase(valid_end, vec.end());
    // Explicitly shrink the capacity to match the size
    vec.shrink_to_fit();
    auto end = std::chrono::high_resolution_clock::now()
    auto elapsed_duration = end - start;
    
    for (const auto& v: vec) std::cout << v << std::endl;
    return 0;
}
```

See, `remove_if(begin_iterator, end_iterator, predicate)` moves all elements that does NOT satisfy the predicate, i.e., eleemtns we want to keep.  

`erase()` will:

- Return iterator pointing to the end of the container. (which is `new_end` in the above snippet)
- Invalidate references and iterators after the new end.
- Change the underlying `size` parameter, of course.

`vec.shrink_to_fit()` reduces the vector capacity to match vector's size.

This is how `remove_if` is implemented: (from [cpp reference](https://en.cppreference.com/w/cpp/algorithm/remove))

```cpp
template<class ForwardIt, class UnaryPred>
ForwardIt remove_if(ForwardIt first, ForwardIt last, UnaryPred p)
{
    first = std::find_if(first, last, p);
    if (first != last)
        for (ForwardIt i = first; ++i != last;)
            if (!p(*i))
                *first++ = std::move(*i);
    return first;
}
```


### profiling results

One interesting finding is that `erase-remove` might not be faster than copying:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>    // for std::remove_if, std::copy_if
#include <chrono>

using namespace std;    // don't be like me in production code, I'm being lazy :)

std::vector<int> get_vec(){
    std::vector<int> vec;
    for (int i = 0; i < 100000; ++i) vec.push_back(i);
    return vec;
}

int main()
{
    constexpr int NUM = 1;
    unsigned int nano_seconds_count = 0;

    for (int i = 0; i < NUM; ++i) {
        auto vec = get_vec();
        auto start = std::chrono::high_resolution_clock::now();
        auto valid_end = std::remove_if(vec.begin(), vec.end(), [](const int& i){ return i > 1000; });
        vec.erase(valid_end, vec.end());
        // Explicitly shrink the capacity to match the size
        vec.shrink_to_fit();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_duration = end - start;
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_duration);

        nano_seconds_count += elapsed_seconds.count();
        std::cout << vec.size() << std::endl;
    }
    std::cout << nano_seconds_count / NUM << " nanoseconds (remove_if + erase)" << std::endl;

    nano_seconds_count = 0;

    for (int i = 0; i < NUM; ++i) {
        auto vec = get_vec();
        std::vector<int> vec_new;
        vec_new.reserve(vec.size());
        auto start = std::chrono::high_resolution_clock::now();
        std::copy_if(vec.begin(), vec.end(), std::back_inserter(vec_new), [](const int& i){ return i <= 1000; });
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_duration = end - start;
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_duration);

        nano_seconds_count += elapsed_seconds.count();
                std::cout << vec_new.size() << std::endl;

    }
    std::cout << nano_seconds_count / NUM << " nanoseconds (copy_if)" << std::endl;

    return 0;
}
```

In the above snippet, we measure the time it takes to remove elements above 1000 from an array `[0, 100000]`. It turns out that `std::copy_if` is 1.25x faster! Why? In this case, we are copying **many fewer elements** than the swapping `remove_if` has to do. So, if you have an application that **filters out a lot of elements**, it might be worth it to try the **copy_if** method. However, in general I would still stick to `erase-remove`, because it performs better in case there are **not many elements** to filter out. (I'm risk-averse)