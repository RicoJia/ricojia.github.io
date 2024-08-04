---
layout: post
title: C++ - Casting 
date: '2023-01-23 13:19'
subtitle: Ever tried convincing your boss that your need for vacation is a 'const' by using const_cast? Welcome to C++ casting!
comments: true
tags:
    - C++
---

## Introduction

TODO - coming soon ;)

## Const Casting

❗ `const_cast` is needed sometimes to form a pointer/reference to a const object to interface with legacy code. Modifying the `char*` might lead to undefined behavior. So be careful.

```cpp
// Initialize a char* array with string literals, using const_cast
char *my_argv[] = {const_cast<char*>("this_test_program"), nullptr};
```

- string literals are of type `const char*`. `const_cast<char*>` is to cast away the constness.
