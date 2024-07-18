---
layout: post
title: C++ Virtual Keyword - Virtual Functions and Virtual Inheritance
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

### Remember to Define Virtual Dtor in Base Class

In the case of virtual dtor, Base and Derived Class can look like below when virtual dtor is / is not defined in base class. (**One weirdity here is if ~Base::Base is marked virtual, Derived::Derived is automatically virtual**)

```cpp
// Base and Derived classes vtables when virtual dtor is defined in base class:
[&Base::do_something, &~Base::Base]
[&Derived::do_something, &~Derived::Derived]

// Base and Derived classes vtables when virtual dtor is not defined in base class:
[&Base::do_something ]
[&Derived::do_something]

// virtual dtor only in base? or in both? 
```

So when `~Base::Base` is not marked virtual, the compiler will first synthesize `~Base::Base`. But then during runtime, the program can't find `~Derived::Derived`, so that's not called. Try running below code snippet:  
```cpp
#include <iostream>
#include <memory>

struct Shape {
    virtual void do_something() const = 0; // Pure virtual function
    // Omitting virtual destructor will skip the dtor of Circle as well.
    // virtual ~Shape() = default;
};

struct Circle : public Shape {
    void do_something() const override {
        std::cout << "circle" << std::endl;
    }
    ~Circle() {
        std::cout << "Circle destructor called" << std::endl;
    }
};

int main() {
    Shape* shape = new Circle();
    shape->do_something();
    delete shape; // Won't see "Circle destructor called"
    return 0;
}
```


### Quirks

- Virtual functions cannot be called in ctor or dtor TODO
TODO: to link, C++ does not allow virtual function calls in constructors and destructors. You can call a method declared as virtual, but it will not use dynamic dispatch. It will use the definition from the class the constructor of which is currently executing. This is because calling a virtual method before the derived class constructor has a chance to run is very dangerous - the virtual method might operate on uninitialized data. Therefore, if you need to call a method that will be overridden in a derived class, you have to use SetUp()/TearDown().

## Virtual Inheritance

The Dreadful Diamond on Derivation Problem

TODO

## References

1. Stack Overflow explanation on Virtual inheritance: stackoverflow.com/a/21607/5652483
2. Wikipedia on Dynamic Dispatch: https://en.wikipedia.org/wiki/Dynamic_dispatch