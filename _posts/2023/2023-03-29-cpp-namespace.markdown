---
layout: post
title: C++ - Namespace
date: 2023-03-18 13:19
subtitle: Anonymous Namespace
comments: true
header-img: img/post-bg-alitrip.jpg
tags:
  - C++
---
# C++ Anonymous Namespaces: The Modern Way to Hide File-Local Code

In C and C++, it is common to write helper functions or file-local variables that should only be used inside a single source file. These helpers are implementation details. They are not part of the public API, and other `.cpp` files should not be able to reference them.

In C, the usual way to do this is with `static`:

```cpp
static int helper_function() {    return 42;}
```

For free functions and global variables, `static` gives the symbol **internal linkage**, meaning it is only visible within that translation unit, usually a single `.cpp` file.

In modern C++, the preferred approach is often an **anonymous namespace**. Everything declared inside `namespace { ... }` is placed in an unnamed, file-local namespace. The effect is similar to using `static`: the names are only visible within that `.cpp` file. An anonymous namespace gives **internal linkage** to the things declared inside it. For example, two `.cpp` files can both have a function named `helper()` inside anonymous namespaces. They do not conflict at link time because each one is local to its own translation unit.

```cpp
namespace {  
  
int helper_function() {  
return 42;  
}  
  
int cached_value = 10;  
  
} // namespace
```

One reason anonymous namespaces are considered the more idiomatic C++ solution for file-local implementation details: You cannot make a class or struct itself `static` in the same way you can make a function `static`.

```cpp
namespace {
 struct ParserState {    int line_number;};
}
```

Anonymous namespaces should generally **not** be used in header files. Because every `.cpp` file that includes the header gets its own separate copy of the declarations inside the anonymous namespace.
