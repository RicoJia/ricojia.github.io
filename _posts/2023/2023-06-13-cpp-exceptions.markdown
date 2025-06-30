---
layout: post
title: Cpp Exceptions
date: '2023-06-13 13:19'
subtitle: Error Throwing
comments: true
header-img: "img/post-bg-infinity.jpg"
tags:
    - Linux
---

## Error Throwing

rethrow an error using `std::rethrow_exception(std::exception_ptr)`

```cpp
catch (std::exception&)
{
    exptr_ = std::current_exception();
}
if (exptr_)
{
    // The official interface of std::exception_ptr does not define any move
    // semantics. Thus, we copy and reset exptr_ manually.
    const auto exptr_copy = exptr_;
    exptr_ = nullptr;
    std::rethrow_exception(exptr_copy);
}
```
