---
layout: post
title: C++ - Useful Macros
date: '2023-01-05 13:19'
excerpt: Why do we still need macros for C++? Please click in and take a look!
comments: true
---

## Introduction

Macros are string substitution for code. It happens during the preprocessing phase. It is generally not encouraged in c++ nowadays. However, there are still applications where it could be handy / necessary

Here is the general compilation process

<div style="text-align: center;">

```mermaid
graph TD;
    A[Source Code main.cpp] --> B[Preprocessing, output main.i];
    B --> D[Compilation output main.s];
    D --> E[Assembly Code, output main.o ];
    E --> H[Linking output executable];
    
    style A fill:#8FBC8F,stroke:#333,stroke-width:2px,color:#fff;
    style B fill:#4682B4,stroke:#333,stroke-width:2px,color:#fff;
    style D fill:#D2691E,stroke:#333,stroke-width:2px,color:#fff;
    style E fill:#8B008B,stroke:#333,stroke-width:2px,color:#fff;
    style H fill:#FF4500,stroke:#333,stroke-width:2px,color:#fff;
```
</div>

## Application 1 - Linkage Specifier

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

