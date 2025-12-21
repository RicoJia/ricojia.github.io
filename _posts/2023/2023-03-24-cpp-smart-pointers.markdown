---
layout: post
title: C++ - [Pointers - 2] - Smart Pointers
date: '2023-03-18 13:19'
subtitle: unique_ptr, shared_ptr
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Unique Pointer

### Basics

```cpp

#include <memory>
std::unique_ptr<int> ptr = std::make_unique<int>(1);
```

- Note that a `unique_ptr` is only `8 bytes` on a 64 bit system, and **it's the same size as a raw pointer**, because it is just a wrapper that ensures sole ownership of the pointer.

## Shared Pointer

### Memory Usage of Shared Pointer

Note that a `shared_ptr` is **2x8=16 bytes** itself. It includes:
    - a raw pointer
    - a pointer to the control block.

The control block has a reference counter, deleter, etc. Its implementation is **platform-dependent**. Roughly, we need 8 bytes for the count, and 8 bytes for the pointer to the deleter. So, 16 bytes.

Can you see the difference here?

```cpp
auto ptr = std::make_shared<Type> (args);
// vs
auto ptr2 = std::shared_pointer<Type> (new Type());
```

- `ptr` has 1 memory allocation call: control block + the object itself
- `ptr2` has 2 memory allocation calls: control block and the object itself separately

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
