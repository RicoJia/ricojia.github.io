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

- Construction
    - Copy Construction
    - Move construction
        - It's in the form of:
            ```cpp
            Meter temp(2);
            Meter m3(std::move(temp));  // This will call move constructor (if implemented correctly)
            ```

    - Note: Copy ellision is a optimzation introduced in C++17 for the scenario where a temporary object is created, and is assigned to a new object.
        ```cpp
        Foo f2 = make_foo();    // Otherwise see: ctor + copy construction
        Meter m3(Meter(4));     // Otherwise see: ctor + move construction
        ```

- Assignment
  - An assignment is to assign an existing object to another object. It returns an instance of itself, so we can do **chain assignment**: `a=b=c`
  - It can be done by **copy assignment** or **move assignment**.
    - Copy assignment
      - Copy assignment just appears to be "copying." It does not copy, it calls constructor directly. It's in the form of `m=m3;`
      - It looks for an `non-explicit` ctor.
    - Move Assignment:
      - It's in the form of `m = Meter(3);`

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
    Meter m3(Meter(4));     // ctor
    Meter m4(std::move(m2)); // Move construction
}
```

## Conversion and `explicit`

- Implicit conversion
    - Single-argument constructor. E.g., if a constructor is:
        ```cpp
        class Foo{
            Foo(double d){}
        }; 
        foo(Foo f){}
        foo(3.0);
        ```
        - `foo(3.0);` implicitly converts `3.0` to Foo.
- Explicit conversion
    - In many scenarios, we do not want such automatic conversion to happen, because they are hidden and might yield unwanted side effects.
    - We first need to prohibit implicit conversion by declaring a constructor `explicit`, then, one can define a conversion operator
        ```cpp
        class Foo{
        public:
            explicit Foo(double d){}
            explicit operator float() const {return 1.0f;} 
        }; 

        float d = Meter(3.0);   // ❌ implicit conversion is banned
        float f = float(Foo(4.0));  // ✅ explicit conversion should be used
        ```
- Notes:
    - Be careful with implicit conversions between types:
        ```cpp
        class Foo{
        public:
            explicit Foo(double d){}
            operator double() const {
                cout<<"double"; 
                return value_;}
            explicit operator float() const {return 1.0f;} 
            double value_=100;
        }; 
        void run(float){}
        int main()
        {
            float f = Foo(4.0); // 👀 This still works! we see "double" in the output
        }
        ```
        - This compiles fine, because **the compiler finds implicit conversions Foo -> double->float**. So though the direct conversion `Foo->float` is banned, if a path can be found, the compiler will still compile.
    - In the following example, despite the existence of a default arg, the keyword `explicit` is still meaningful because it prohibits implicit conversion.
        ```cpp
        class ICP3D {
        public:
            explicit ICP3D(const Options& options = Options()) {}
        };
        void run(ICP3D icp) {}

        Options opt;
        run(opt);   // Not gonna work, because implicit conversion is banned.
        run(ICP3D(opt));
        ```
