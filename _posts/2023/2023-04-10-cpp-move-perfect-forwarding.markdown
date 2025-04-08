---
layout: post
title: C++ - Move Semantics, Perfect Fowarding
date: '2024-04-10 13:19'
subtitle: Universal Reference
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - ROS
---

## Universal Reference And Perfect Forwarding

Perfect fowarding in C++ preserves the type and value categories of **an input argument**.

Refresher - Value categories:
- lvalue
- rvalue

In the following snippet, we have a few interesting aspects

```cpp
template<typename Func>
void profile_and_call(Func&& func){
    std::forward<Func>(func)();
}
```

### Universal Reference

`std::forward` calls a callable be it a lvalue, or an rvalue. When a type is deduced in a template, if it's in the form `&&`, it's a universal reference, `Func&&` in this case is the **universal reference** to `func`.  So if it's:

- an lvalue: `profile_and_call(another_function)`, `another_function` is `another_function&`, and `Func&&` is deduced to `another_function& &&` and becomes `another_function&`
- an rvalue: `profile_and_call(std::move(another_function))`, `another_function` is `another_function&&`, and `Func&&` is deduced to `another_function&& &&` and becomes `another_function&&`

Specifically, this preserves the value category of the function, specifically when the function inside has move semantics. If it's a simple `const Func& func` argument, we will not be able to move. Also, if the function is a captured lambda, it's a temporary object that will be best used if it's "moved"


On the other hand, note that **T&& is a universal reference only when T is a deduced template parameter.** (so it doesn't happen outside of the template scenario).

```cpp
void func(int&& arg);   // rvalue reference only
```




