---
layout: post
title: C++ - [Compilation 5] Compilation Speed Ups
date: '2023-05-13 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Header-only Library Hurts Compilation Time

My halo library takes ~2-3min to compile. I have ~20 executables, each executable does `target_include_files(70_header_files)`. 

Why It’s Slow

- Every translational unit is compiled once. The library src files are compiled first, then the user executables. 
- Each executable compiles with its own `#include halo/xxx.hpp`. If `halo/xxx.hpp` includes many other header files, they will be dragged in as well. 
    - Only `compiled object code in` `.cpp` goes files into the archive `libhalo.a`. Header files are not compiled. **This is why header only hurts**
    - So the time cost is `size_of_include_graph * number of affected TU * single_recompilation_time`. 


- Templates make it worse 
    - Template declaration and definition must be in include files.  Whenever a translation unit (a .cpp file) uses (ODR-uses) `MyTemplate<T>`, the compiler implicitly instantiates that specialization in that TU, and each instantiation costs compile time.
    - If we just need one instance for all source files, we can declare the instance in one cpp file, then mark it as extern in the header file.
        ```cpp
        // In MyTemplate.hpp
        template<typename T>
        struct MyTemplate { void foo(); /*…*/ };
        extern template struct MyTemplate<int>;    // “don’t instantiate here”

        // In MyTemplate.cpp
        #include "MyTemplate.hpp"
        template struct MyTemplate<int>;           // instantiate once
        ```

### What Can Be Done?

- Move heavy include, implementations into cpp. 
- Pimpl (Cheshire Cat) idiom.
    - Strip implementation details out of public headers entirely: public API headers contain only a forward-declared struct Impl; pointer.
    - Changes in private implementation no longer force recompilation of dependents.

- [Advanced] Unity (Jumbo) builds. CMake has the `UNITY_BUILD` target property.  By grouping multiple small .cpp into a single “unity” TU, you amortize include overhead.
    ```cpp
    set_target_properties(halo PROPERTIES UNITY_BUILD ON)
    ```

-  [Advanced] PreCompiled HEaders (PCH), where `pch.hpp` includes the heaviest, most-stable headers.

    ```cpp
    target_precompile_headers(halo
    PRIVATE
        <Eigen/Dense>
        "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/include/halo/pch.hpp>"
    )
    ```

