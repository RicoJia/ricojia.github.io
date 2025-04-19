---
layout: post
title: C++ - Datatypes
date: '2023-01-05 13:19'
subtitle: std::optional, structural binding, iostream, namespace
header-img: "img/post-bg-alitrip.jpg"
comments: true
tags:
    - C++
---


## POD Types

- char or unsigned char is 1 byte
- float, int, are usually 4 bytes (32 bits) on a 64 bit machine
- double is 8 bytes (64 bits)
- long double is 8, or even 16 bytes

### Common Bugs

- Observed Error:  error: no matching function for call to ‘min(float, double)’

```cpp
// Ensure both arguments are of type float
rk_deltas_[i] = std::min(my_var, 2.0f);
```

## Optional

`std::optional` can specify if a value is valid or not. It has:

- `.has_value()` to examine if it has value
- `.value()` to retrieve the value

```cpp
#include <optional>
#include <iostream>
#include <vector>
using namespace std;

void test_optional(){
    auto get_ll = [](bool does_return)-> std::optional<size_t> {
        if(does_return) 
            return 1;
        else 
            return {};
    };
    cout<<get_ll(true).value()<<endl;
    if (get_ll(false).has_value()){
        cout<<"yay"<<endl;
    }
}

int main(){
    test_optional();
}
```

## Namespace

- `::testing::InitGoogleTest(&argc, argv);`: defensive way to avoid accidental namespacing issue. This is to call the gtest module


## Structural Binding
TODO

## `iostream`

c means "character-based stream" in `cin`, `cout`.

- `cout`: This is the standard output stream and **is buffered by default**. This means that data sent to cout is stored in a buffer and is written out (flushed) either **when the buffer is full**, when a newline is encountered (depending on the implementation and whether the stream is tied to an input stream), or when it is explicitly flushed (using `std::flush` or `std::endl`).

- `cerr`: This stream is meant for error messages and **is unbuffered by default**, **so it flushes its output immediately**. This is useful when you want to ensure error messages are output right away.

- `clog`: Similar to cerr in that it’s used for logging messages, but unlike cerr, clog is buffered. This means it collects log messages in a buffer and flushes them less frequently, which can improve performance.

Because clog is buffered, **it can be more efficient for logging non-critical messages** since it reduces the number of flush operations. Downside: if your program crashes before the buffer is flushed, some log messages might be lost. 

There is clog as well. `clog` is faster because it stores output to file / screen in a buffer, once the buffer is full, the buffer flushes. In contrast, `cout` flushes immediately. `clog` is most useful in writing non-essential logs that could be lost upon system crashes.