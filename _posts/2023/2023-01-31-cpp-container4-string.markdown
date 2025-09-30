---
layout: post
title: C++ [Container 4] Strings
date: '2023-01-29 13:19'
subtitle: std::string_view
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## `std::string_view`

`std::string_view` is a non-owning view of a character sequence. It doesn't have dynamic initialization or ctor, and it's trivially copyable (always 16 bytes that include ptr and length). It can be initialized from string literals **during compile time**

```cpp
constexpr std::string_view str = "hello";
static_assert(str.size() == 14);
```

- No lifetime management, no allocation
- `std::string` is always null-terminated, but `std::string_view` not required to be.
- Usage: `std::string_view` is **read-only.**
- Comparison:

```cpp
std::string_view(m.name) = name;    // name is std::string_view, m.name is std::string
```
