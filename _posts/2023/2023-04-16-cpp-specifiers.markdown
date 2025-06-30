---
layout: post
title: C++ - Specifiers
date: '2024-04-16 13:19'
subtitle: Declaration-Only Specifiers
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - ROS
---

## Declaration Only Specifiers

These specifiers will only appear in an `.hpp` file, not in `.cpp` files.

| Keyword                    | What it does                                                    |
| -------------------------- | --------------------------------------------------------------- |
| `virtual`                  | Marks a member as polymorphic                                   |
| `override`                 | Verifies you’re actually overriding a base-class virtual method |
| `final`                    | Prevents further overrides or inheritance                       |
| `= 0` (pure virtual)       | Makes a method abstract                                         |
| `= default` / `= delete`   | (De)generates or disables special member functions              |
| `explicit`                 | Forbids implicit conversions in constructors                    |
| `inline`                   | Allows multiple definitions (headers) without ODR conflicts     |
| `constexpr` / `consteval`  | Enables compile-time evaluation                                 |
| Default arguments (`= 42`) | Gives callers a default value                                   |
| `template`                 | Begins a template declaration                                   |
| `friend`                   | Grants access to private members                                |

Repeated Specifiers in source code
