---
layout: post
title: C++ - Functions, Lambda
date: '2023-02-13 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Lambda Basics

Lambda functions are introduced when? 

Generic lambda (cpp14+) can take in an arbitrary type

One example of lambda is when working with lots of `memcpy`
    ```
    You can use a generic lambda (available since C++14) to make copyToBuffer work with any type:

    size_t pos = 0;
    auto copyToBuffer = [&](const auto& variable) {
        memcpy(buffer + pos, &variable, sizeof(variable));
        pos += sizeof(variable);
    }
    ```
