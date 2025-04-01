---
layout: post
title: C++ - [Memory Access - 1] Layout
date: '2023-05-27 13:19'
subtitle: Cache
comments: true
header-img: "img/post-bg-infinity.jpg"
tags:
    - C++
---

## Memory Access: The Hidden Bottleneck

[Reference](https://www.akkadia.org/drepper/cpumemory.pdf)

## Why Memory Access Matters

Even as CPU speeds continue to improve, the cost of fetching data from memory has become a limiting factor. Operating systems mitigate this by caching frequently used data in RAM — which is much faster than disk. However, we can'replace SSDs or HDDs with RAM: **RAM is expensive, and it loses content when power is lost**. So RAM is usually called "memory".

To have fast memory access while still retaining disks, we rely on hardware-accelerated memory access, such as:

- Fast, parallel RAM architectures
- Advanced memory controller designs
- **CPU caches (L1, L2, L3)**
- **Direct Memory Access (DMA) for device communication**

The memory hierarchy is:

```
CPU Registers   →   L1 Cache   →   L2 Cache   →   L3 Cache   →   RAM   →   Disk
   (tiny, fast)      ↑              ↑             ↑             ↑         ↑
   (nanoseconds)  Cache for...  Cache for...  Cache for...  Cache for...  (slow)
```

## CPU Caches: Your Program's Real Performance Friend

CPU caches are inside CPU. They are even faster than RAM. Their size typically is 64KB - 64 MB. Its latency is ~10ns, compared to 100ns to RAM. Caches work by keeping frequently accessed data close to the processor:
- L1: Smallest, fastest, per-core.
- L2: Bigger, a bit slower.
- L3: Shared across cores, larger still.

A **cache line** is typically **64 bytes** — **the smallest unit of data** fetched from cache. So reading arr[0] (with 4-byte integers) will likely bring in arr[0] through arr[15].

<div style="text-align: center;">
    <p align="center">
       <figure>
            <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/39393023/428782551-796d79f8-7704-4990-9c21-f40b13bb2e7a.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250331%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250331T220634Z&X-Amz-Expires=300&X-Amz-Signature=aa504fb90d19cc694074edde3b1bcb0462e11bce580af366111f6ec9164a7be0&X-Amz-SignedHeaders=host" height="300" alt=""/>
       </figure>
    </p>
</div>

Caches are organized into sets and associativity levels:

- For example, in a 4-way set-associative cache, each set holds 4 cache lines.
- Memory addresses are mapped to these sets — which leads to a few interesting behaviors.


### Cache Miss Types

A cache miss happens **when the data the CPU needs is not in the cache — so it has to go fetch it from a slower memory** (like RAM or even disk).

- Cold Miss: First-time access.

- Conflict Miss: Two memory addresses map to the same cache set. If that set is full, one must be evicted — even if the rest of the cache is empty.

- Capacity Miss: The working set is too large to fit in the cache.

### False Sharing: An Invisible Type Of Multithreading Slowdowns

False sharing occurs when multiple threads access different variables that share a cache line. Even though each thread is logically isolated, the CPU sees contention due to hardware cache coherence protocols.

Example: 

```cpp
struct Counters {
    int a; // Thread 1
};
std::vector<SafeCounter> counters(num_threads);
```

If `counters[0], counters[1]` are on the same cache line, updates from different threads will invalidate the cache line across cores, causing unnecessary slowdowns. (The result of the final result will be correct.)

```cpp
struct alignas(64) SafeCounter {
    int a; // Thread 1
};

// In C
struct __attribute__((aligned(64))) SafeCounterC {
    int counter;
};

std::vector<SafeCounter> counters(num_threads);
```
The `alignas(64)` will make sure **the start of the struct is aligned to a 64-byte boundary** (the size of a typical cache line). In a small test with 2 thredads, assigning counter incrementing with `alignas(64)`took 3ms, while without it the program took 12ms. 

TL;DR: **Align and pad data structures to avoid false sharing!**