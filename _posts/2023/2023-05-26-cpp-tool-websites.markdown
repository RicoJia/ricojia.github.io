---
layout: post
title: C++ Tool Websites
date: '2023-05-26 13:19'
subtitle: CppInsights, Compiler Explorer, Conan
comments: true
header-img: "img/post-bg-infinity.jpg"
tags:
    - C++
---

## CppInsights

CppInsights does these things:

- how the standard library objects (like stringstream, vector, etc.) get instantiated 
- how overloaded operators are actually resolved.


```cpp
std::vector<int> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9};
std::vector<int> sub = {4, 5, 6};
auto it = std::search(vec.begin(), vec.end(), sub.begin(), sub.end());

std::list<int> lst = {10, 20, 30, 40, 50, 60};
std::list<int> sub_lst = {30, 40};
auto it_2 = std::search(lst.begin(), lst.end(), sub_lst.begin(), sub_lst.end());
```

Becomes:

```cpp
std::vector<int, std::allocator<int> > vec = std::vector<int, std::allocator<int> >{std::initializer_list<int>{1, 2, 3, 4, 5, 6, 7, 8, 9}, std::allocator<int>()};
std::vector<int, std::allocator<int> > sub = std::vector<int, std::allocator<int> >{std::initializer_list<int>{4, 5, 6}, std::allocator<int>()};
__gnu_cxx::__normal_iterator<int *, std::vector<int, std::allocator<int> > > it = std::search(vec.begin(), vec.end(), sub.begin(), sub.end());
std::list<int, std::allocator<int> > lst = std::list<int, std::allocator<int> >{std::initializer_list<int>{10, 20, 30, 40, 50, 60}, std::allocator<int>()};
std::list<int, std::allocator<int> > sub_lst = std::list<int, std::allocator<int> >{std::initializer_list<int>{30, 40}, std::allocator<int>()};
std::_List_iterator<int> it_2 = std::search(lst.begin(), lst.end(), sub_lst.begin(), sub_lst.end());
```

## Compiler Explorer
TODO

## Conan

[Conan is a Python-based package manager written by JFrog.](https://github.com/conan-io/conan) It works on recipes (small tool scripts), and can install packages from [the Conan Center](https://conan.io/center). 

Workflow:

1. Specify packages to install in `conanfile.txt`
2. Conan calculates the package id based on compiler version, OS, arch, build type (release), etc.
3. Check local cache; Try download from Conan, or build from source
4. In CMakeLists.txt, add below

    ```bash
    # Use Conan's generated toolchain file (this sets compiler flags, paths, etc.)
    include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
    include(${CMAKE_BINARY_DIR}/conan_deps.cmake)

    find_package(MYPACKAGE REQUIRED)
    ```
5. Build