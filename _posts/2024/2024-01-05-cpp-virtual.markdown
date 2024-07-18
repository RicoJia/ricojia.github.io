---
layout: post
title: [C++] Virtual Keyword - Virtual Functions and Virtual Inheritance
date: '2024-01-05 13:19'
excerpt: Virtual is virtually complicated. Dynamic Dispatch, Dreadful Diamond Derivative (DDD) Problem ...
comments: true
---

## Introduction

The virutal keyword is very versatile. A really short and nasty summary is, it is to ensure the correct functions or base class is inherited / loaded during runtime.

## Virtual Functions

In Computer science, **dispatching** is the action of selecting which polymorphic function to call [2]. In C++, there are static dispatching (like templates) and dynamic dispatching (polymorphism). In static dispatching, the dispatched function is known during compile time, whereas in dynamic dispatching, the dispatched function won't be known until the runtime type of an object is known. This way, polymorphism could become useful when we have different types of objects with inheritance, and want to call their corresponding versions of a function.

The mechanism of dynamic dispatching in C++ is through a virtual function table (vtable), which stores pointers to all virtual functions' addresses. Vtable will only be looked up when polymorphism is used.

<p align="center">
<img src="https://github.com/user-attachments/assets/204338dc-a36d-4dfc-80a9-d0dcba31eec2" height="400"/>
<figcaption><a href="https://pabloariasal.github.io/2017/06/10/understanding-virtual-tables/">Source: Pablo Arias</a></figcaption>
</p>

By declaring a function `virtual` in base class, one can provide `overrride` functions in subclasses if necessary. One example is:

```cpp
#include <iostream>
#include <string>

class B{
    public:
    virtual void foo() {  std::cout << "Hello, I'm B "<<std::endl;}
    };
class C: public B{
    public:
    void foo() override {  std::cout << "Hello, I'm C"<<std::endl;}
    };
class D: public B{
    };
    
int main()
{
    // See Hello, I'm C
    C().foo();
    // See Hello, I'm B 
    D().foo();
}
```

### If You Have A Base Class, Always Define Virtual Dtor it

In C++ polymorphism, **IT IS REQUIRED TO DECLARE BASE CLASS'S DTOR AS VIRTUAL**

```cpp
#include <iostream>
#include <memory>

struct Shape {
    // Omitting virtual destructor will skip the dtor of Circle as well.
    virtual ~Shape() { std::cout << "Deleting shape" << std::endl;};
};

struct Circle : public Shape {
    ~Circle() {
        std::cout << "Circle destructor called" << std::endl;
    }
};

int main() {
    Shape* shape = new Circle();
    delete shape; // Won't see "Circle destructor called"
    return 0;
}
```

Because this way, the Vtable is able to include the proper destructors. 

```bash
[&~Base::Base]
[&~Derived::Derived]
```

If the base class doesn't have a virtual dtor: 

```cpp
#include <iostream>
#include <memory>

struct Shape {
    // Omitting virtual destructor will skip the dtor of Circle as well.
    ~Shape() { std::cout << "Deleting shape" << std::endl;};
};

struct Circle : public Shape {
    ~Circle() {
        std::cout << "Circle destructor called" << std::endl;
    }
};

int main() {
    Shape* shape = new Circle();
    delete shape; // Won't see "Circle destructor called"
    return 0;
}
```

- **One weirdity here is if ~Base::Base is marked virtual, Derived::Derived is automatically virtual**

### Quirks

- **Virtual functions cannot be called in ctor or dtor**
    - This is because the object is **considered to have the type of base class during construction, or destruction**

## Virtual Inheritance

The Dreadful Diamond on Derivation Problem

TODO

## References

1. Stack Overflow explanation on Virtual inheritance: stackoverflow.com/a/21607/5652483
2. Wikipedia on Dynamic Dispatch: https://en.wikipedia.org/wiki/Dynamic_dispatch