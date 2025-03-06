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

Use parallelisim on associative operations `a+(b+c) = (a+b) + c`

- `std::execution::par`: This policy tells the algorithm to run in parallel. However, while tasks are distributed across threads, the order in which operations within each thread (or chunk) are performed is still relatively predictable. The overall reduction order is unspecified between chunks, but within each chunk it maintains a certain order.

- `std::execution::par_unseq`: This policy not only allows parallel execution but also permits vectorization. Vectorization may rearrange the order of operations even more freely for performance reasons. Because of this extra level of reordering, the final result can differ when using non-associative operations.
    - There's no guarantee that this will be multi-threaded, it could be SIMD instructions only.
