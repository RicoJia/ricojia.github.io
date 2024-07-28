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

<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/dcdabf60-46de-4054-a987-98aa4a28d569" height="400" alt=""/>
    </figure>
</p>

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

