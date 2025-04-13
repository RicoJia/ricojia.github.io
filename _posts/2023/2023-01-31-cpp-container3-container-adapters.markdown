---
layout: post
title: C++ - [Container 3] Container Adaptors
date: '2023-01-29 13:19'
subtitle: `std::queue`, `std::stack`
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

`std::queue`, `std::stack`, `std::priority_queue` are NOT sequence containers, instead, they are wrappers around `std::deque`, and `std::vector` to provide a restricted interface. (FIFO for queue, LIFO for stack). The common interface we examine are:

- `push`, `pop`
- `top`, `front`
- No iterators or allow traversals. 