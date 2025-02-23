---
layout: post
title: C++ - Functions, Lambda
date: '2023-02-13 13:19'
subtitle: Lambda, Functors
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Functor

A functor is an object that acts like a function, using the `overloaded operator()`.

``` cpp
struct greaters{
    greaters(){}
    operator()(int a, int b){return a > b;}
}

struct increment{
private:
int a
public:
    increment(int a){a = a;}
    operator()(int b){return b+a}
}

std::transform(arr, arr+n, arr, increment(a));      //equivalent to calling increment(a) first.
std::make_heap(arr, arr+n, greaters());         //calls constructor first, then inside the function, it will call the overloaded().
```

## Lambda Basics

Lambda functions are introduced when? 

Generic lambda (cpp14+) can take in an arbitrary type, with `auto`

One example of lambda is when working with lots of `memcpy`

```cpp
// You can use a generic lambda (available since C++14) to make copyToBuffer work with any type:

size_t pos = 0;
auto copyToBuffer = [&](const auto& variable) {
    memcpy(buffer + pos, &variable, sizeof(variable));
    pos += sizeof(variable);
}
```

Generic Lambda also works with different input types, due to type deduction:

```cpp
auto add = [](auto a, auto b) { 
    std::cout << b << std::endl;  // Print second argument
    return a;                     // Return first argument
};
add(1, std::string("123"));  // Works fine!
```

## Lambda Capture with Initializer (C++ 14)

`[idx = 0]` is the initializer value in for each. 

```cpp
#include <stdio.h>
#include <vector>
#include <algorithm>
int main()
{
    std::vector<int> vec(10, 0);
    std::for_each(vec.begin(), vec.end(), [idx=0](int& i)mutable{i = ++idx;});
    printf("%d", vec.at(9));
    return 0;
}
```

- what type is idx? 
    - Here **type is deduced**, and `idx` is `int`. we can do: `[idx = 3.14]` (float), `[idx = std::string("hello")]` (a string)
- By default, lambda capture is **const**. We need **mutable** to make sure we can modify the vector
- We can pass a reference **int& i** into the lambda
- `std::for_each` is in `<algorithm>`

## Full Template Lambda (C++20)

In a full template lambda, we can specify the input types **more explicitly**.

```cpp
int main() {
    auto multiply = []<typename T>(T a, T b) {
        return a * b;
    };
    std::cout << multiply.operator()<double>(3, 2.5) << "\n"; // 3 is int. If we specify double, it will be implicitly casted. 7.5  
    std::cout << multiply(3.0, 2.5) << "\n"; // 7.5
}
```

- Like a regular lambda, we can invoke an instance of the template lambda with the synthesized `operator()`.

One limitation is a template lambda can only be stored as a template parameter `template <typename Func>`, not as `std::function<int(double, double)>`: 

```cpp
// Works:
std::function<int(double, double)>  multiply = [](auto a, auto b) {
    return a * b;
};
// Doesn't work:
std::function<int(double, double)>  multiply = []<typename T>(T a, T b) {
    return a * b;
};
```

One slightly longer example is as follows: we want to be able to wrap a function in a generic function for evaluation:

```cpp
#include <iostream>
#include <vector>
#include <utility>

// Example function that times execution
template <typename Func>
void evaluate_and_call(Func&& func, const std::string& name, int times) {
    std::cout << "Running " << name << " " << times << " times...\n";
    for (int i = 0; i < times; ++i) {
        func();  // Execute the function
    }
}

struct T3{};

int main() {
    std::vector<std::pair<size_t, size_t>> matches;

    // Generic lambda + template lambda
    auto lambda = [&]<typename T3>(
                      T3& grid, auto& matches) {
        // grid.GetClosestPointForCloud(first, second, matches);
    };

    T3 grid0;
    evaluate_and_call(
        [&]() { lambda(grid0, matches); }, // so we skipped binding 
        "Grid0 单线程", 
        10
    );

    return 0;
}
```

- `[&]() { lambda(grid0, matches); }` is a smart way to bind different types of grid with the lambda function. This is an `inline, parameter less` function