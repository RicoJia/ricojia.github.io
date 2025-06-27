---
layout: post
title: C++ - [Concurrency 1] Multithreaded Programming
date: '2023-06-01 13:19'
subtitle: `std::thread`, mutex, atomic_variables
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Threading

### Motivation: Why C++ Concurrency Matters

Standardized concurrency arrived with **C++11**, replacing the ad-hoc C / compiler-specific APIs used before C++98 with a formal memory model.

Key library components:

```
std::thread   // spawn & manage threads  
std::unique_lock  // RAII-based mutex ownership  
std::atomic   // lock-free atomics, fences, memory-order flags
```

- These components originated from **Boost’s experimental threads library**; Boost later aligned with the ISO STL.
- RAII underpins safe resource management across threads.
- A well-defined memory model plus low-level atomic primitives enables predictable synchronization.


The difference between **synchronity**, **single-threaded asynchrony**, and **multithreading**

| Model                            | Core Idea                                                          | Typical Unit        | Everyday Analogy                                                                        |
| -------------------------------- | ------------------------------------------------------------------ | ------------------- | --------------------------------------------------------------------------------------- |
| **Synchronous**                  | Caller blocks until work completes                                 | –                   | Set timer → cook egg → *wait* → set timer → boil water                                  |
| **Asynchronous (single thread)** | Start work, then do something else before awaiting result          | **Task**            | Set timer → cook egg → *while waiting* start timer → boil water → await both            |
| **Multithreaded**                | Multiple workers execute in parallel, optionally using async tasks | **Worker / Thread** | Cook A sets timer → cooks egg while doing other chores; Cook B boils water concurrently |

### Context Switching

Even a single-core CPU achieves quasi multithreading via task (context) switching—rapidly swapping thread state on one core.

Multi-core systems still rely on **context switches** to juggle **more runnable threads than available cores** (e.g., OS background services, browsers, editors).

### Process vs Multithreading

| Aspect                 | **Processes**                                                                                      | **Threads**                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Isolation / Safety** | Separate address spaces → strong OS protection → harder for one process to corrupt another.        | Share the same address space → easier data sharing but greater risk of accidental corruption.                |
| **Communication**      | Inter-process communication (IPC) is explicit and often slow (pipes, sockets, shared memory).      | Direct access to shared data; no built-in protection, so mutexes/atomics required.                           |
| **Overhead**           | Higher cost to create, schedule and manage; context switch is heavier.                             | Cheaper to spawn and switch; but each thread still consumes stack (often \~1 MB) and other kernel resources. |
| **Scalability**        | Can distribute across machines (e.g., Erlang “actor” model).                                       | Limited to one machine; too many threads exhaust RAM and scheduler efficiency—thread pools mitigate this.    |
| **When to choose**     | Favor when memory safety/isolation outweighs IPC cost or when you need multi-machine distribution. | Favor when fast, low-latency sharing is critical and the complexity of synchronization is acceptable.        |

Rule of thumb: **Don’t default to multiprocess or multithread—measure** which model yields net benefit for your workload.


### Invariant
> An invariant is a statement that must always be true for a data structure (before & after each operation).

In the context of multithreading, one example is to insert into a lock-free list. Some invariants are:

- Read `prev` and `next` pointers that must still satisfy `prev->next == next`.

- Attempt `CAS(prev->next, next, newNode)`.
    - If another thread changed `prev->next`, the invariant you relied on is gone → retry.
    - On success, fix `newNode->next` and ensure `next->prev` eventually points back.

## Race Condition

Definition – The final value (or overall outcome) could be different depending on the relative timing / ordering of two or more threads that access the same data. 

Race conditions are likely NOT replicable under **a debugger**, or adding log statements. In those cases, debugging points or the additional log statements could mess up the timing required for the race conditions to happen. So they are also called a **Heisenbug**.

Mitigation checklist

- Share immutable data where possible.
- Keep mutex-protected data private; **expose safe accessors** instead of **raw addresses.**
- Use `std::atomic` or `higher-level concurrent containers` when the cost of a lock is too high.

To overcaome the race condition, 

| Technique                     | How it works                                                                                                                                                                              | Strengths                                                                      | Weaknesses                                                                                        |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **Mutex / lock**              | One thread at a time enters a critical section. (`std::mutex`, `std::unique_lock`)                                                                                                        | Simple, general; easy to reason about invariants.                              | Blocking & potential contention; risk of deadlock.                                                |
| **Lock-free algorithms**      | Combine atomic read-modify-write ops (CAS, fetch-add) in retry loops; threads never block.                                                                                                | Scales under contention; avoids kernel scheduling latency.                     | Harder to design & verify; starvation possible.                                                   |
| **Transactional Memory (TM)** | Group reads/writes into a *transaction*. If another core touches the same cache lines, the transaction **aborts** and restarts. (`std::atomic<T>` extensions or hardware TM on some CPUs) | Declarative “all-or-nothing” style—no explicit locks; good for coarse updates. | Limited hardware/compiler support; throughput drops under heavy conflicts due to repeated aborts. |

## Mutex

Here is an example of multi-threading with a detached thread (which will be terminated when the main thread finishes). Note that one could join a thread as well.  

```cpp
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

void worker(std::vector<int>& nums, std::mutex& mtx)        // runs on its own thread
{
    std::cout << "[worker] started\n";

    for (int n : nums) {
        std::lock_guard<std::mutex> lg(mtx);
        std::cout << "  • " << n << "² = " << n * n << '\n';
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cout << "[worker] done\n";
}

int main()
{
    std::vector<int> data{1, 2, 3, 4, 5};
    std::mutex mtx;

    // ───────────────────────────────────────────────
    // 1. Thread constructor spawns the OS thread.
    // 2. The new thread begins executing *immediately*.
    // ───────────────────────────────────────────────
    std::thread t(worker, std::ref(data), std::ref(mtx));  // pass the vector by value (safe)
    // why would you need ref?

    t.detach();                   // fire-and-forget -- we can’t join later
    // t.join();                  // If the thread is not detached, join it. 

    std::cout << "[main] continues\n";

    // Give the worker time to finish so we can watch the output.
    // In real code you’d use join(), condition variables, futures, etc.
    std::this_thread::sleep_for(std::chrono::seconds(2));
}
```



Some subtleties about `std::thread` include:

- When the `std::thread` constructor is called it **decay-copies** every argument into an **internal tuple** so the new thread has **its own copy**.
    - “Decay-copy” means: **strip references, cv-qualifiers, and array/function types**, then copy or move the resulting value.
    - This is why raw reference wouldn't work, and `std::ref` is used.






















