---
layout: post
title: C++ - Smart Pointers
date: '2023-03-18 13:19'
subtitle: unique_ptr, shared_ptr
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Common Operations

Reset: `std::unique_ptr::reset()` and `std::shared_ptr::reset`

- `unique_ptr::reset()` is basically basically `delete ptr; ptr = nullptr`
- `std::shared_ptr::reset()` can be a bit slower, because of deallocation & allocation of referecen count.

```cpp
std::unique_ptr<int> ptr;
ptr.reset(new int(1));
std::shared_ptr<int> s_ptr;
s_ptr = std::make_shared<int>(3);
s_ptr.reset(new int(3));
```

Ownership:

- `unique_ptr` is move-assigned.

```cpp
// ownership
// unique_ptr is move-assigned
std::unique_ptr<int> new_ptr = std::make_unique<int>(5);
// If the count reaches 0, it also deletes the object and the control block
std::shared_ptr<int> s_ptr2 = std::make_shared<int>(6);
s_ptr = s_ptr2; // now the element 3 is deleted since it's not referenced
```

## Conversions

`unique_ptr` -> `shared_ptr`: use `std::move` to transfer ownership

```cpp
std::unique_ptr<Foo> u = std::make_unique<Foo>();

std::shared_ptr<Foo> s = std::move(u);    // OK
// or:
std::shared_ptr<Foo> s2(std::move(u));    // OK
```
