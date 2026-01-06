---
layout: post
title: C++ - [Concurrency 2] Conditional Variables
date: 2023-06-02 13:19
subtitle: Conditional Variables, Jthread
comments: true
header-img: img/post-bg-unix-linux.jpg
tags:
  - C++
---

## Conditional Variables

### Vanilla Usage

```cpp
std::unique_lock<std::mutex> lock(mutex_);
cv_.wait(lock, pred);

do_stuff_1();
do_stuff_2();
```

What actually happens

1. `std::unique_lock` locks the mutex.
2. `cv_.wait(!pred())` basically:

```cpp
   std::unique_lock<std::mutex> lk(mutex);
   
   while (!pred()){
    unlock(mutex);    
  sleep_on_cv(mutex);// wakes on notify_one/all or spuriously
  lock(mutex)
   }
   // here it's still locked
   do_stuff_1(); 
   ```

## Jthread (C++20)

The main characteristics of `std::jthread` include:

- Can join itself automatically, or can request to join using `std::stop_token`
- The thread is `movable but not copyable`
- `jthread` can still be controlled using the older style `t.join()` and `t.detach()`

```cpp
#include <thread>
#include <iostream>

// Simple function
void task() {
    std::cout << "Hello from jthread!\n";
}

// Function that accepts stop_token
void cancellable_task(std::stop_token stoken) {
    int i = 0;
    while(!stoken.stop_requested()) {
        std::cout << i++ << " ";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main() {
    // Basic usage - auto-joins
    std::jthread t1(task);
    
    // With stop token
    std::jthread t2(cancellable_task);
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    t2.request_stop();  // Request cancellation. non blocking
    std::cout<<"terminated"<<std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // Both threads automatically joined when they go out of scope. 
    // If no request_stop is issued, the thread will be called with std::stop_token automatically. 
    // Thread destruction at the end will be blocking 
}
```

In a more advanced case, one can control when to stop multiple threads using `std::stop_source`, from an arbitrary thread:

```cpp
#include <thread>
#include <iostream>
#include <chrono>

void worker(std::stop_token stoken, int id) {
    while (!stoken.stop_requested()) {
        std::cout << "Worker " << id << " running\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "Worker " << id << " stopped\n";
}

int main() {
    // Create a stop_source manually
    std::stop_source source;
    
    // Create multiple threads that share the same stop mechanism
    std::jthread t1(worker, source.get_token(), 1);
    std::jthread t2(worker, source.get_token(), 2);
    std::jthread t3(worker, source.get_token(), 3);
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Stop all threads at once using the shared source
    std::cout << "Requesting stop for all workers\n";
    source.request_stop();
    
    // All threads will stop and auto-join when they go out of scope
}
```

    - `std::stop_token` must be used with `std::stop_source`, but `std::stop_source` could be passed around as reference or value. 
