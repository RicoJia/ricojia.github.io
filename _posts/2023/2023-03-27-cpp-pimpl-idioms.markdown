---
layout: post
title: C++ - Pimpl Idiom
date: '2023-03-18 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Minimal Structure

```cpp
// header

// Widget.hpp
#pragma once
#include <memory>

class Widget {
public:
  Widget();
  ~Widget();

  void doSomething();

  // (Rule of five if you need copy/move; omitted for brevity)

private:
  struct Impl;
  std::unique_ptr<Impl> pImpl;
};
```

In source file, we can have all the implementation in here to reduce recompilation-time upon changes (since this is a compilation unit and its code is compiled once per change). 

```cpp
// Widget.cpp
#include "Widget.hpp"
#include <iostream>

// 1) Define the Impl struct in the .cpp so its details stay hidden:
struct Widget::Impl {
  void doSomethingImpl() {
    std::cout << "Widget::Impl doing work\n";
  }
};

// 2) Forward all Widget methods to Impl:
Widget::Widget() : pImpl(std::make_unique<Impl>()) {}
Widget::~Widget() = default;

void Widget::doSomething() {
  pImpl->doSomethingImpl();
}
```

## Is pimpl worth it??

I forgot the return keywrod. Then I was stuck at the assocaited `abort` for 1h+. So please be aware that pimpl boiler plate could be a source of bugs

```cpp
NavState IEKFLIO::IEKFLIO::get_current_state() const {
    impl_->get_current_state();
}
`