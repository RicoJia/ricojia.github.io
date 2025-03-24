---
layout: post
title: C++ - [Concurrency 2] Coroutine
date: '2023-02-20 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Coroutine

Coroutines in C++ allows us to write asynchronous code in a sequential style. It allows us to resume and suspend, instead of blocking a thread. A coroutine by default is single-threaded, but they can be manually scheduled to run on other threads. C++20 is just the beginning of coroutines, honestly, its programming is not intuitive... It needs a large infrastructure library to rival languages like `go` for server programming

For example, if you resume a coroutine in a separate thread, you can choose to resume it on a separate thread.

TODO: 
[Reference](https://gqw.github.io/posts/c++/coroutine/)