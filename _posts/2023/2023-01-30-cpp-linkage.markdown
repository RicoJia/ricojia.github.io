---
layout: post
title: C++ - Linkage
date: '2023-01-30 13:19'
subtitle: In C++, Linkage is Either External Or Internal
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---
## Linkage

In C and C++, linkage determines the accessibility of functions & variables defined in one source file in other different translation units (i.e., different source files).

There are three types of linkages: internal, external Linkage Linkage, and no linkage. The no linkage case is trivial: those are local variables defined in functions. Below we will focus on global variable sharing.

## Vanilla Case - Global Variable Definition In Header

Usually, we can share global variables with a single variable declared and defined in a header file. These satisfies most simple cases.

```cpp
// foo.hpp
#pragma once
#include <iostream>

int num = 0;    // Easy Peasy

void num_inc();

// foo_funcs.cpp
#include "foo.hpp"

void num_inc(){
    num++;
    std::cout<<"num: "<<num<<std::endl;
}

// foo.cpp
#include "foo.hpp"

int main(int argc, char** argv){
    num_inc();
}
```

### Disadvange Of The Vanilla Case

The vanilla case, however, is prone to the **"multiple definition error"**. That is, what if we have more than 1 variable with the same type and name defined? C++ requires that in a program, only 1 variable can be defined across all translational units (**One Definition Rule, ODR**). 

E.g.,

```cpp
// foo.hpp
int num = 0;

// foo.cpp
int num=0;  // ODR violation: it's been defined in foo.hpp!!
```

In many cases, it's a common mistake. However, there could be times when:

- We want to define the variable in a translation unit
- Or define it in header file, but a translational decides to use its own copy

Below we will walk through these two cases.

### External Linkage

One alternative is to use external linkage. We declare a variable in `.hpp` as `extern`, then define it in one (and only one) source file.

```cpp
// foo.hpp
extern int num;
// foo_funcs.cpp
int num = 0; 
// foo_main.cpp
std::cout<<"main, num: "<<num<<std::endl;
```

People claim that this is helpful for declaring variables in modules. However, I find declaring in header files (the vanilla method) more intuitive. Both methods would give the multiple definition errors if more than two definitions of the same variables are found.

### Internal Linkage

Another alternative is internal linkage. This method allows us (and forces us if necessary) to instantiate a separate copy of the variable in a source file.

```cpp
// foo.hpp
#pragma once
#include <iostream>

void num_inc();

// foo_main.cpp
int num = 100;

// foo_funcs.cpp
static int num = 0; 
```

Without the static keyword, the compiler will throw a multiple definition error: `multiple definition of 'num'; /tmp/ccKTJUBw.o:(.data+0x0): first defined here`.

With the static keyword, `foo_funcs.cpp` creates its own copy of `num`. This technique could be useful for creating isolating environments (e.g., for testing).