---
layout: post
title: C++ - [OOP] Polymorphism Substitute - Curiously Recurring Template Pattern (CRTP)
date: '2023-03-03 13:19'
subtitle: static polymorphism
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

### Optimization Impact: ⭐️⭐️⚪⚪⚪

CRTP is a design pattern where the base class takes in derived class as a template parameter. This way, no virtual function is created, no runtime cost is spent on vtable look up, and **static polymorphism** is achieved.

```cpp
template <typename Derived>
class Addable {
    Derived operator+(const Derived& other) const {
            return Derived(static_cast<const Derived*>(this)->get() + other.get());
    }

};


class MyInt : public Addable<MyInt> {
    int value;
public:
    MyInt(int v) : value(v) {}
    int get() const { return value; }
};

class MyFloat : public Addable<MyInt> {
    float value;
public:
    MyFloat(int v) : value(v) {}
    int get() const { return value; }
};
```

For more benchmarking, [please check this code snippet and try yourself!](https://godbolt.org/z/Ta73x33G8). Without any optimization, CRTP is 103ms, while runtime polymorphism is 170ms. But with -O3 optimization, both are 0ms. So the optimization effect is limited. 