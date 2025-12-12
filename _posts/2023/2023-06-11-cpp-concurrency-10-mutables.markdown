---
layout: post
title: C++ - [Concurrency 10] Mutable  Variables
date: '2023-06-01 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Mutable Variable

In this example, we are able to change a mutable variable in a `const` function

```cpp
#include <iostream>
#include <string>

class Foo {
public:
    Foo(std::string name) : name_(std::move(name)) {}

    // This function is logically "const" (it doesn't change the observable state
    // of the object), but we still want to keep a call counter.
    void do_something() const {
        // allowed because cnt_frame_ is mutable
        ++cnt_frame_;

        std::cout << "Foo(" << name_ << ") do_something called "
                  << cnt_frame_ << " times\n";
    }

    std::size_t call_count() const {
        return cnt_frame_;
    }

private:
    std::string name_;
    mutable std::size_t cnt_frame_ = 0;  // can be modified in const functions
};
```
