---
layout: post
title: C++ - [OOP] Destruction
date: '2023-03-02 13:19'
subtitle: Destruction Ordering
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

The destruction order is `Derived Class -> Derived class members -> base class`, which is the inverse order of construction: `base class -> Derived class members -> Derived Class`

```cpp
#include <iostream>

struct A
{
    ~A() { std::cout << "A\n"; }
};

struct C
{
    ~C() { std::cout << "C\n"; }
};

struct B : public A
{
    C c1;                   // data-member declared *after* any implicit A sub-object

    ~B() { std::cout << "B\n"; }
};

int main()
{
    {
        B obj;              // construct a B on the stack
    }                       // scope ends → destructors run

    return 0;
}
```
