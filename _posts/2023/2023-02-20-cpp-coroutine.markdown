---
layout: post
title: C++ - Coroutine
date: '2023-02-20 13:19'
subtitle: SIMD
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Coroutine

Coroutines in C++ allows us to write asynchronous code in a sequential style. It allows us to resume and suspend, instead of blocking a thread. A coroutine by default is single-threaded, but they can be manually scheduled to run on other threads. 

For example, if you resume a coroutine in a separate thread, you can choose to resume it on a separate thread.

```cpp

```