---
layout: post
title: C++ - [Concurrency 8] `std::this_thread::yield()`
date: '2023-06-01 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Introduction

`std::this_thread::yield()` is used in a case where you are busy waiting and fighting for CPU time slices. It will give the scheduler a hint that other threads could come in and finsih its time slice (so this thread may or may not relinquish CPU). If there are other waiting threads, its resultant wait is still very short and uses a lot more CPU than `std::this_thread::sleep()`.

## Bench marking

```cpp
// cpu_yield_vs_sleep_bench.cpp
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

using clock_tp = std::chrono::steady_clock::time_point;

struct Result {
    double wall_seconds{};
    double cpu_seconds{};
    double cpu_util_percent{};
    double iters_per_sec{};
    unsigned long long iters{};
};

Result run_yield_test(double seconds) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::clock_t c0 = std::clock();

    unsigned long long iters = 0;
    while (true) {
        std::this_thread::yield();
        ++iters;

        if ((std::chrono::steady_clock::now() - t0) >= std::chrono::duration<double>(seconds))
            break;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const std::clock_t c1 = std::clock();

    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double cpu  = double(c1 - c0) / CLOCKS_PER_SEC;

    Result r;
    r.wall_seconds = wall;
    r.cpu_seconds = cpu;
    r.cpu_util_percent = (wall > 0) ? (cpu / wall) * 100.0 : 0.0;
    r.iters = iters;
    r.iters_per_sec = (wall > 0) ? iters / wall : 0.0;
    return r;
}

Result run_sleep_test(double seconds, std::chrono::nanoseconds sleep_ns) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::clock_t c0 = std::clock();

    unsigned long long iters = 0;
    while (true) {
        std::this_thread::sleep_for(sleep_ns);
        ++iters;

        if ((std::chrono::steady_clock::now() - t0) >= std::chrono::duration<double>(seconds))
            break;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const std::clock_t c1 = std::clock();

    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const double cpu  = double(c1 - c0) / CLOCKS_PER_SEC;

    Result r;
    r.wall_seconds = wall;
    r.cpu_seconds = cpu;
    r.cpu_util_percent = (wall > 0) ? (cpu / wall) * 100.0 : 0.0;
    r.iters = iters;
    r.iters_per_sec = (wall > 0) ? iters / wall : 0.0;
    return r;
}

void print_result(const std::string& name, const Result& r) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== " << name << " ===\n";
    std::cout << "Wall time (s):        " << r.wall_seconds << "\n";
    std::cout << "CPU time (s):         " << r.cpu_seconds << "\n";
    std::cout << "CPU utilization (%):  " << r.cpu_util_percent << "\n";
    std::cout << "Iterations:           " << r.iters << "\n";
    std::cout << "Iters/sec:            " << r.iters_per_sec << "\n\n";
}

int main(int argc, char** argv) {
    // Defaults
    double duration_sec = 3.0;
    long long sleep_ns = 1'000'000; // 1 ms

    // Simple CLI:
    //   cpu_yield_vs_sleep_bench [duration_sec] [sleep_ns]
    if (argc >= 2) duration_sec = std::stod(argv[1]);
    if (argc >= 3) sleep_ns = std::stoll(argv[2]);

    std::cout << "Benchmark duration: " << duration_sec << " s\n";
    std::cout << "Sleep interval:     " << sleep_ns << " ns\n\n";

    // Warm-up to stabilize scheduling
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto ry = run_yield_test(duration_sec);
    print_result("this_thread::yield()", ry);

    auto rs = run_sleep_test(duration_sec, std::chrono::nanoseconds(sleep_ns));
    print_result("sleep_for(" + std::to_string(sleep_ns) + " ns)", rs);

    std::cout << "Note:\n"
              << " - On lightly loaded systems, yield() often shows very high CPU% (busy-wait style),\n"
              << "   whereas sleep_for() should be near 0% CPU during the sleep intervals.\n"
              << " - Actual numbers depend on OS scheduler, timer resolution, and system load.\n";
    return 0;
}
```

Results:

- Sleep for 1ms in 3s

```
=== this_thread::yield() ===
Wall time (s):        3.000
CPU time (s):         2.996
CPU utilization (%):  99.875
Iterations:           14030597
Iters/sec:            4676865.278

=== sleep_for(1000000 ns) ===
Wall time (s):        3.000
CPU time (s):         0.038
CPU utilization (%):  1.280
Iterations:           2786
Iters/sec:            928.601
```

- Sleep for 100us in 5s

```
=== this_thread::yield() ===
Wall time (s):        5.000
CPU time (s):         5.000
CPU utilization (%):  99.992
Iterations:           22570644
Iters/sec:            4514128.715

=== sleep_for(100000 ns) ===
Wall time (s):        5.000
CPU time (s):         0.129
CPU utilization (%):  2.584
Iterations:           32438
Iters/sec:            6487.599
```

- Sleep for 10ms in 2s

```
=== this_thread::yield() ===
Wall time (s):        2.000
CPU time (s):         1.999
CPU utilization (%):  99.926
Iterations:           9036116
Iters/sec:            4518057.790

=== sleep_for(10000000 ns) ===
Wall time (s):        2.005
CPU time (s):         0.006
CPU utilization (%):  0.321
Iterations:           198
Iters/sec:            98.763
```

## When is `std::this_thread::yield()` useful?

1. In spinlocks or busy wait loops (when response time is in microseconds). `std::this_thread::yield()` prevents other threads from starving.

```
std::atomic_flag lock = ATOMIC_FLAG_INIT;

void lock_spin() {
    while (lock.test_and_set(std::memory_order_acquire)) {
        std::this_thread::yield();  // let others progress
    }
}

void unlock_spin() {
    lock.clear(std::memory_order_release);
}
```
