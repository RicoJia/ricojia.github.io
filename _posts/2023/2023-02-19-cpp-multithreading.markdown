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

### `std::transform(std::execution::par_unseq)`

```cpp
#include <algorithm>
#include <execution>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec1(1000, 1);
    std::vector<int> vec2(1000, 2);
    std::vector<int> vec3(1000, 0);

    // Using parallel unsequenced execution policy
    std::transform(std::execution::par_unseq, vec1.begin(), vec1.end(),
                   vec2.begin(), vec3.begin(),
                   [](int a, int b) { return a + b; });

    // Verify the first few results
    for (int i = 0; i < 10; i++) {
        std::cout << vec3[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
```

- If we need index, do:

```cpp
std::vector<size_t> indices(vec1.size());
std::iota(indices.begin(), indices.end(), 0);
```

## `std::async`:

```cpp
#include <iostream>
#include <future>
#include <vector>
#include <limits>

// A simple structure to hold task results.
struct AlignResult {
    bool success;
    double cost;
};

// Dummy alignment task: returns success if the input is even,
// and the cost is simply the input value.
AlignResult alignTask(double value) {
    AlignResult result;
    result.success = (static_cast<int>(value) % 2 == 0);
    result.cost = value;  
    return result;
}

int main() {
    // Define some input values for our tasks.
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    std::vector<std::future<AlignResult>> futures;
    
    // Launch each alignment task in a separate thread.
    for (double val : values) {
        futures.push_back(std::async(std::launch::async, [val]() {
            return alignTask(val);
        }));
    }
    
    // Retrieve results and choose the one with the lowest cost.
    double bestCost = std::numeric_limits<double>::max();
    bool found = false;
    for (auto &fut : futures) {
        AlignResult res = fut.get();
        std::cout << "Task result: success=" << res.success << ", cost=" << res.cost << std::endl;
        if (res.success && res.cost < bestCost) {
            bestCost = res.cost;
            found = true;
        }
    }
    
    if (found) {
        std::cout << "Best cost: " << bestCost << std::endl;
    } else {
        std::cout << "No successful alignment found." << std::endl;
    }
    
    return 0;
}
```
-  Each value in values is processed by `alignTask` in its own thread using `std::async(std::launch::async, ...)`. `std::launch::async` launches new threads. Some implementations might optimize by reusing threads, but this behavior is not guaranteed.
