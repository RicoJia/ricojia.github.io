---
layout: post
title: Common Robotics CPP Software Packages
date: '2023-02-01 13:19'
subtitle: TBB
comments: true
tags:
    - Robotics
---

## TBB

TBB (Intel Threading Building Blocks) is a parallel-programming library that abstracts out details for managing threads, synchronization, or load balancing explicitly. Key features include:

- Task parallelism
- Data parallelism
- Concurrent Data Structures

Example usage: parallelize a for loop

```cpp
#include <tbb/parallel_for.h>
#include <vector>

int main(){
    // initialize the vector to 1000 1s 
    std::vector <int> vec{1000, 1};
    tbb::parallel_for(tbb::blocked_range<size_t>(0, data.size()),
        [&](const tbb::blocked_range<size_t>& range){
            for(size_t i = range.begin(); i != range.end(); ++i){
                data[i] += i;
            }
        }
    )
}
```

To install:

```cpp
sudo apt-get update
sudo apt-get install libtbb-dev
```

To link:

```cpp
cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Find the TBB package
find_package(TBB REQUIRED)

# Add the executable target
add_executable(MyExecutable main.cpp)

# Link against the TBB::tbb target
target_link_libraries(MyExecutable PRIVATE TBB::tbb)

# (Optional) Specify C++ standard
set_target_properties(MyExecutable PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
```
