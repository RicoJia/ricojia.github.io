---
layout: post
title: C++ - [Compilation 2] Compiler Directives
date: '2023-05-04 13:19'
subtitle: Macros, Pragmas, Attributes
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Introduction

Compiler directives are instructions that are not part of C++ / C standards, but tell the compiler how to compile the code. They starts with `#` (pronounced as "hash"), **and are handled in the pre-processing phase and before the actual compilation**.

- `#include`
- `#pragmas` pragmas - compiler-specific instructions
- `#define` macros
- attributes

## Macros

Macros are string substitution for code. It happens during the preprocessing phase. It is generally not encouraged in c++ nowadays. However, there are still applications where it could be handy / necessary

Here is the general compilation process


<div style="text-align: center;">

<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/dcdabf60-46de-4054-a987-98aa4a28d569" height="400" alt=""/>
    </figure>
</p>

</div>

### Application 1 - Linkage Specifier

```cpp
#ifdef __cplusplus
    #define CV_IMPL extern "C"
#else
    #define CV_IMPL
#endif

CV_IMPL void cvFindExtrinsicCameraParams2(...){
    cvConvertPointsHomogeneous(objectPoints, matM);
    cvConvertPointsHomogeneous(imagePoints, _m);
}
```

This is effectively `extern "C" void cvFindExtrinsicCameraParams2(...){}` and `void cvFindExtrinsicCameraParams2(...){}`.

- `extern "C"` is a "linkage specifier". In C++, function names are mangled so they will be unique in the object code. E.g., `foo()` could become `_fooid1()`. This is also the underlying mechanism for function overloading. However, C does not have function overloading. So, when linking C code in C++, we need to make sure names of function symbols are unique, and exclude name mangling.

## Pragmas

[GCC pragmas](https://gcc.gnu.org/onlinedocs/gcc/Pragmas.html)

- `#pragma GCC / #pragma clang / #pragma warning`: control warnings, optimizations
    - Diagnostic pragmas:
        ```cpp
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wpedantic"
        #include "sophus/se3.hpp"
        #pragma GCC diagnostic pop  // recover diagnostic directive 
        ```
- `#pragma once`: ensuring a header is included once
- `#pragma pack(n)` - control data alignment.

## Attributes / Annotations

Some compilers allow attributes to be added to functions / variables / types:

- `__attribute__((...))` in GCC/Clang
- `__declspec(...)` in MSVC