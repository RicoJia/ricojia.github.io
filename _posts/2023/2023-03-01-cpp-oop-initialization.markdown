---
layout: post
title: C++ - [OOP] Initialization
date: '2023-01-15 13:19'
subtitle: Default Initialization
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Default Initialization

Default Initialization is **when a variable is initialized without explicit value**. They could be in an uninitialized state and accessing them could be an undefined behavior.

- POD types like `int`, `float` are initialized to an **intermediate value**. Accessing them without initialization is an undefined behavior,
- User defined types should either **have no constructor provided, or have a default constructor present.**

```cpp
class A{
    A(){};  // This is NOT default initialization
    A() = default;  // This IS default initialization
};

class B{
    int value;  // There's a synthesized default initializer as well.
};
```

- `static` storage duration always default initializes to 0, whereas automatic storage variables are default initialized to intermediate values

```
int x;          // automatic storage, intermediate value
static int y;   // default initialized to 0
int *j;         // automatic storage, intermediate value
static int *p   // static storage, default initialized to nullptr

int arr[5];        // Elements have indeterminate values
static int arr[5]; // All elements default-initialized to 0
```

Best Practices:

- Use **brace initialization** `{}` (C++ 11) to explicitly initialize variables

```cpp
int i{};    // explicitly initialized to 0;
```