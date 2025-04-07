---
layout: post
title: C++ - [Concurrency 1] Various Concurrent Methods
date: '2023-02-19 13:19'
subtitle: Vectorization, `std::launch`, Lockless array writes, OpenMP, SIMD
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Asynchronous Programming

- Single-Threaded Asynchornous Programming
    - Cooperative Multitasking & Pre-emptive Multitasking
        - This is a model where each task **voluntarily gives up control** after executing X instructions, so other tasks can execute. This is in contrast to the **preemptive multitasking, where the scheduler forcibly interrupts tasks**.
            - Say a line-printing thread prints 3 lines then yield control.
    - Event loop 
        - Used in Like Javascript in a browser / `Node.js`
        - The python `Asyncio` has an event loop that schedules and runs coroutines.
        ```
        Event Loop:
            while queue.get():
                event = queue.pop_front(); // FIFO
                event.handler(args)
        ```
        - Event could be OS notification mechanisms, like epoll, select, poll.

- Concurrent Programming
    - This involves multiple threads / processes running in parallel. (True parallelism)

## Summary of Performance

- SIMD is fast because it processes several data elements at once with minimal overhead. **THIS IS EXTREMELY USEFUL IN COMPUTER VISION** where one may find many "small loops"
- Vectorization using `std::execution` policies can yield similar performances as SIMD, which is great
- Raw For loop might be faster than you think
- OpenMP might be good for large, independent iterations but might not pay off for small loops due to thread management overhead.

Here I have two tasks and they both show the trend above:

    <div style="text-align: center;">
        <p align="center">
        <figure>
                <img src="https://github.com/user-attachments/assets/eb9a0a9a-ed42-4b81-a3a5-f106426e7099" height="300" alt=""/>
        </figure>
        </p>
    </div>

    <div style="text-align: center;">
        <p align="center">
        <figure>
                <img src="https://github.com/user-attachments/assets/a8df0176-3b42-40db-9caa-4c75cd796f1c" height="300" alt=""/>
        </figure>
        </p>
    </div>

## Code For SIMD, OpenMP, Vectorization, and Raw For Loop(Benchmark) Comparisons

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <execution>
#include <chrono>
#include <smmintrin.h> // SSE4.1 intrinsics for SIMD
#ifdef _OPENMP
#include <omp.h>
#endif

// g++ -O2 -std=c++17 omp_test.cpp -ltbb
void test_vectorization(std::vector<int>& data, int N){
    data.assign(N, 0);
    // Initialize vector with indices 0, 1, 2, ..., N-1.
    std::iota(data.begin(), data.end(), 0);

    auto start = std::chrono::high_resolution_clock::now();

    std::for_each(std::execution::par_unseq, data.begin(), data.end(), [](int &i) {
        // Compute the expression.
        int val = (i * i) / 3 + 12;
        // If even, subtract 1.
        if (val % 2 == 0)
            val = val - 1;
        i = val;
    });

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed (test_vectorization): " << elapsed.count() << " seconds.\n";
}

// g++ -O2 -msse4.1 -std=c++17 omp_test.cpp
void test_SIMD(std::vector<int>& data, int N){
    data.assign(N, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int i = 0;
    for (; i <= N - 4; i += 4) {
        // Create a vector of indices.
        __m128i idx = _mm_set_epi32(i+3, i+2, i+1, i+0);
        // Convert indices to float.
        __m128 float_idx = _mm_cvtepi32_ps(idx);
        // Compute (i * i) in float.
        __m128 float_square = _mm_mul_ps(float_idx, float_idx);
        // Divide by 3.0f.
        __m128 div = _mm_div_ps(float_square, _mm_set1_ps(3.0f));
        // Add 12.0f.
        __m128 add = _mm_add_ps(div, _mm_set1_ps(12.0f));
               // Convert result back to integers.
        __m128i result = _mm_cvtps_epi32(add);
        // Check for evenness: even numbers have LSB = 0.
        __m128i one = _mm_set1_epi32(1);
        __m128i even_mask = _mm_cmpeq_epi32(_mm_and_si128(result, one), _mm_setzero_si128());
        // Create a subtract mask: if even, subtract 1; else subtract 0.
        __m128i subtract = _mm_and_si128(even_mask, one);
        __m128i new_result = _mm_sub_epi32(result, subtract);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&data[i]), new_result);
    }
    // Process any remaining elements in scalar code.
    for (; i < N; i++) {
        int val = (i * i) / 3 + 12;
        if (val % 2 == 0)
            val = val - 1;
        data[i] = val;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed (test_SIMD): " << elapsed.count() << " seconds.\n";
}

void test_omp(std::vector<int>& data, int N){
    data.assign(N, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < N; ++i) {
        int val = (i * i) / 3 + 12;
        if (val % 2 == 0)
            val = val - 1;
        data[i] = val;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed (test_omp): " << elapsed.count() << " seconds.\n";
}

void test_benchmark(std::vector<int>& data , int N){
    data.assign(N, 0);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < N; ++i) {
        int val = (i * i) / 3 + 12;
        if (val % 2 == 0)
            val = val - 1;
        data[i] = val;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed (test_benchmark): " << elapsed.count() << " seconds.\n";
}

int main() {

    std::vector<int> data;
    for (int N = 1000; N < 1e11; N *= 10){
        //TODO
        std::cout<<"============ N: "<<N<<" ============"<<std::endl;
        test_vectorization(data, N);
        test_SIMD(data, N);
        test_omp(data, N);
        test_benchmark(data, N);
    }
    
    // Print first 10 results to verify.
    for (int i = 0; i < 10; ++i)
        std::cout << "data[" << i << "] = " << data[i] << "\n";
    return 0;
}
```

- To compile and run: `g++ -fopenmp -O2 -msse4.1 -std=c++17 omp_test.cpp -ltbb && ./a.out`
    - `msse4.1` is for SIMD, `-fopenmp` is for OpenMP

------------------------------------------------------------
## [Method 1] SIMD
See above example

## [Method 2] Lockless Vector Writes - Vert Powerful and Practical

When using a pre-allocated vector in C++, you're effectively working with a contiguous block of memory in your process's virtual address space. Once allocated, the addresses of the vector's elements remain fixed—even though the underlying operating system manages physical memory mapping behind the scenes. This means that **if each thread writes exclusively to different elements of the vector, there's no risk of memory reallocation or shifting during execution, ensuring thread-safe writes without locks.**

[Referece](https://stackoverflow.com/questions/45720829/can-two-threads-write-to-different-element-of-the-same-array)

```cpp
#include <iostream>
#include <vector>
#include <thread>

constexpr size_t NUM_THREADS = 4;
constexpr size_t ARRAY_SIZE = 16;

void worker(int thread_id, std::vector<int>& vec, size_t start, size_t end) {
    for (size_t i = start; i < end; ++i)
        vec[i] = thread_id;
}

int main() {
    // Pre-allocate the vector in virtual memory.
    std::vector<int> vec(ARRAY_SIZE, -1);
    std::vector<std::thread> threads;
    size_t chunk = ARRAY_SIZE / NUM_THREADS;

    for (size_t i = 0; i < NUM_THREADS; ++i) {
        size_t start = i * chunk;
        size_t end = (i == NUM_THREADS - 1) ? ARRAY_SIZE : start + chunk;
        threads.emplace_back(worker, i, std::ref(vec), start, end);
    }

    for (auto& t : threads) {
        t.join();
    }

    // Print the vector to verify that each element was updated by the correct thread.
    for (int v : vec)
        std::cout << v << " ";
    std::cout << std::endl;
    return 0;
}
```

------------------------------------------------------------
## [Method 3] Vectorization Execution Policy `std::execution::par_unseq` (C++ 17)

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

- `std::execution::par`: This policy tells the algorithm to run in parallel. However, **while tasks are distributed across threads**, the order in which operations within each thread (or chunk) are performed is still relatively predictable. The overall reduction order is unspecified between chunks, but within each chunk it maintains a certain order.

- `std::execution::par_unseq`: This policy not only allows parallel execution but also permits vectorization. Vectorization may rearrange the order of operations even more freely for performance reasons. Because of this extra level of reordering, the final result can differ when using non-associative operations.
    - There's no guarantee that this will be multi-threaded, it could be SIMD instructions only.

### `std::transform(std::execution::par_unseq)`

```cpp
std::vector<int> vec1(1000, 1);
std::vector<int> vec2(1000, 2);
std::vector<int> vec3(1000, 0);

// Using parallel unsequenced execution policy
std::transform(std::execution::par_unseq, vec1.begin(), vec1.end(),
                vec2.begin(), vec3.begin(),
                [](int a, int b) { return a + b; });
```

- If we need index, do:

```cpp
std::vector<size_t> indices(vec1.size());
std::iota(indices.begin(), indices.end(), 0);
```

### Dependency on TBB

The parallel algorithms in your STL implementation (like when using `std::execution::par_unseq`) sometimes use TBB as a backend. You might see:

```
undefined reference to tbb::detail::r1::execution_slot(tbb::detail::d1::execution_data const*)'
```

To link: do `g++ -O2 -std=c++17 omp_test.cpp -ltbb` (note, libraries like `-ltbb` appear after the filename)


------------------------------------------------------------
## [Method 4] `std::async`

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

-  Each value in values is processed by `alignTask` in its own thread using `std::async(std::launch::async, ...)`. `std::launch::async` launches new threads and **immediately start them**. Some implementations might optimize by reusing threads, but this behavior is not guaranteed.
- Alternatively, one can use `std::async(std::launch::deferred)`, function execution will be **synchronous** and single-threaded. They will start when `future.get()` is called. 

------------------------------------------------------------
## [Method 5] OpenMP

See above example