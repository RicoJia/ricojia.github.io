---
layout: post
title: C++ - Numeric
date: '2023-01-20 13:19'
subtitle: NaN
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## `std::numeric`

### NaN: (C++11)

- `std::numeric_limits<float>::quiet_NaN()`: propagates through arithmetic operations without triggering floating-point exceptions.

```cpp
float nan_value = std::numeric_limits<float>::quiet_NaN();

// see nan
std::cout << "NaN value: " << nan_value << std::endl;

// Checking if a value is NaN
if (std::isnan(nan_value)) {
    std::cout << "The value is NaN." << std::endl;
}
// Propagation example
float result = nan_value + 5.0f;  // Still NaN
std::cout << "Result of NaN + 5: " << result << std::endl;
```

- `std::numeric_limits<T>::signaling_NaN`: signaling NaN (SNaN) can be used to throw a SIGPIPE signal, which quits (crashes) the program. However, with a signal handler, we can convert the sigal to an exception, that can be caught 

```cpp
#include <iostream>
#include <limits>
#include <csignal>
#include <cfenv>
#include <stdexcept>

// Global function to handle SIGFPE and throw an exception
void signal_handler(int signal) {
    if (signal == SIGFPE) {
        throw std::runtime_error("Floating-point exception: signaling NaN encountered!");
    }
}

int main() {
    // Register SIGFPE handler to throw an exception
    // It must have a signal handler, otherwise, this program would just silently crash
    try {
        // Register SIGFPE handler to throw an exception
        std::signal(SIGFPE, signal_handler);

        // Enable floating-point exceptions for invalid operations
        feenableexcept(FE_INVALID);

        // Create a signaling NaN
        double snan = std::numeric_limits<double>::signaling_NaN();

        std::cout << "Attempting to use signaling NaN..." << std::endl;

        // Perform an operation that will trigger SIGFPE
        double result = snan + 1.0;

        std::cout << "This won't execute: " << result << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    return 0;
}
```