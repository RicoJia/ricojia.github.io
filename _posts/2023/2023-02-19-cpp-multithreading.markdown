---
layout: post
title: C++ - Multithreading
date: '2023-02-19 13:19'
subtitle: SIMD
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## SIMD & Vectorization Execution Policy for `for_each` (C++ 17)

`std::execution::par_unseq` is an execution policy introduced in C++17 that you can pass to algorithms like `std::for_each`. It directs the algorithm to execute in parallel and in an unordered fashion, allowing the implementation to use both multi-threading and vectorization (SIMD). This means that **iterations may be run concurrently without any guarantee of order**, so you must ensure that your loop body is free of data races and side effects that depend on ordering.

- Note: different execution policies have their own types. So:

```cpp
// We have to use an if-else because the parallelism_policy are different types
// auto parallelism_policy = parallel ? std::execution::par_unseq : std::execution::seq;
if (parallel) {
    std::for_each(std::execution::seq, matches.begin(), matches.end(), 
        [idx = 0](NNMatch& match) mutable {
            match.idx_in_this_cloud = idx++;
        });
} else {
}
```
