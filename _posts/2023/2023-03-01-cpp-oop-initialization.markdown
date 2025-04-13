---
layout: post
title: C++ - [OOP] Initialization 1 - Construction Basics
date: '2023-03-01 13:19'
subtitle: Conversion, Assignment, Copy
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Constructor And Assignment

When an new object is created, it's either:

- Constructed from scratch
- Or copied from another object. The object could be temporary or permanent.

Conversion and Explicit? 


- Assignment
    - An assignment is to assign an existing object to another object. It can be done by **copy assignment** or **move assignment**. 
    - It returns an instance of itself, so we can do **chain assignment**: `a=b=c`
    - Copy assignment
        - Copy assignment just appears to be "copying." It does not copy, it calls constructor directly. It's in the form of `Meter m = 10;`
        - It looks for an `non-explicit` ctor. 

- How do I see move? 
Meter temp(2);
Meter m3(std::move(temp));  // This will call move constructor (if implemented correctly)

```cpp
#include <iostream>
#include <list>
using namespace std;

class Meter {
    double value_;
public:
    // The big 5 since C++ 11: 
    // ctor, copy ctor, move ctor, copy assignment, move assignment
    // Ctor
    Meter(const double& m): value_(m) { 
        std::cout << "ctor\n"; 
    }
    // Copy Ctor
    Meter(const Meter& other):  value_(other.value_){
        std::cout << "copy ctor\n"; 
    }
    // Move ctor
    Meter(Meter&& other): value_(std::move(other.value_)){
        std::cout << "move ctor: value = " << value_ << "\n";
    }

    // Copy assignment, usually this is synthesized
    Meter& operator=(const Meter& other) {
        std::cout << "copy assignment\n";
        value_ = other.value_;
        return *this;
    }

    Meter& operator=(Meter&& other){
        std::cout << "move assignment\n";
        value_ = std::move(other.value_);
        return *this;
    }

};
int main()
{
    Meter m(1); // ctor
    Meter m2(m);    // copy ctor
    m = Meter(3);   // ctor + move assignment
    m=m2;   // copy assignment
    // Copy ellision even with -O0, C++ 17, to optimize into one ctor call.
    Meter m3(Meter(2)); // I see ctor, not move ctor?
}
``

