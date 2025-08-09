---
layout: post
title: C++ - chrono
date: '2023-03-18 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Basic Timing: high-resolution clock

```cpp
#include <chrono>

void foo(){
    auto start = std::chrono::high_resolution_clock::now();
    std::cout<<"Hello World";
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> duration_ns = end - start;
    std::chrono::duration<double, std::micro> duration_us = end - start;
    std::chrono::duration<double, std::milli> duration_ms(start-end);
    std::chrono::duration<double> duration_s(start-end);

    std::chrono::duration<double, std::ratio<60>> duration_min = end - start;
    std::chrono::duration<double, std::ratio<3600>> duration_hr = end - start;
    std::cout << "Took " << duration_s.count() << " ms\n";  // see Hello WorldTook -1.2644e-05 ms
}
```

- Here, we get `duration_X` in double. By default, `std::chrono::duration<TYPE>` returns a **long int**
- If we accidentally called `start-end`, we see negative time value.
- Notice that below seconds, we have `std::nano`, etc. Above seconds, we use `std::ratio`.

### Time Conversion To `ns, us, s, hour`

**Use `std::duration_cast`**. It will return **a `long int`.**

```cpp
void print_in_different_units( std::chrono::duration<double, std::milli> duration){
    long int ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    long int us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    long int ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    long int s = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    long int m = std::chrono::duration_cast<std::chrono::minutes>(duration).count();
    long int h = std::chrono::duration_cast<std::chrono::hours>(duration).count();
    std::cout << "Took " << ns << " ns\n";
}
```

- Or, use `chrono_literals`
  - It's better to use `using namespace std::chrono_literals;`, then `timer_ = create_wall_timer(10ms, std::bind(&VectorProducer::control_cycle, this));`
