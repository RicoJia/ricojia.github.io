---
layout: post
title: C++ - [OOP] Polymorphism - Virtual Functions and Virtual Inheritance
date: '2023-03-03 13:19'
subtitle: Virtual is virtually complicated. Dynamic Dispatch, Dreadful Diamond Derivative (DDD) Problem ...
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Introduction

The virutal keyword is very versatile. A really short and nasty summary is, it is to **ensure the correct functions or base class is inherited / loaded during runtime**.

## Motivating Example

Virtual functions in C++ allow derived classes to override methods from a base class, enabling runtime polymorphism. This means that when you call a virtual function through a pointer or reference to the base class, the correct derived implementation is executed. In the example below, the class **VertexVelocity inherits from g2o::BaseVertex**. By overriding the read function, it provides a specialized behavior instead of relying on the base implementation:

```cpp
class VertexVelocity : public g2o::BaseVertex<3, Vec3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity() {}

    // Overriding the old functions
    virtual bool read(std::istream& is) { return false; }
}
```

## Public, Protected and Private Inheritance

C++ gives the user control for accessing parent classe's data. E.g.,

```cpp
class B{

public:
    int a;
protected:
    int b;
private:
    int c;
};

class A : protected B{
int get_a() const {return a;}   // good, a is a protected member of A
int get_b() const {return b;}   // good, b is also a protected member of A
int get_c() const {return c;}   // bad, private members in B are never accessible to child classes
};
```

There are 3 types of inheritance. From parent class to child class, access mapping is as follows:

- public: `public -> public`, `protected -> protected`
- protected : `public -> protected`, `protected -> protected`
- private: `public -> private`, `protected -> private`

**Private members of the parent class are never accessible to child classes** , directly. But if the parent class has public functions that invokes private functions, then yes, indirectly a child class can access them.

## Virtual Functions

In general computer science, **dispatching** is the action of selecting which polymorphic function to call [2]. In C++, there are **static dispatching (like templates)** and **dynamic dispatching (polymorphism)**. In static dispatching, **the dispatched function is known during compile time**, whereas in dynamic dispatching, the dispatched function **won't be known until the runtime type of an object is known**. This way, polymorphism could become useful when we have different types of objects with inheritance, and want to call their corresponding versions of a function.

The mechanism of dynamic dispatching in C++ is through a virtual function table (vtable), which stores pointers to all virtual functions' addresses. Vtable will only be looked up when polymorphism is used.

<p align="center">
<img src="https://github.com/user-attachments/assets/204338dc-a36d-4dfc-80a9-d0dcba31eec2" height="400"/>
<figcaption><a href="https://pabloariasal.github.io/2017/06/10/understanding-virtual-tables/">Source: Pablo Arias</a></figcaption>
</p>

By declaring a function `virtual` in base class, one should provide `overrride` functions in subclasses if they want to apply polymorphism on the function. One example is:

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

### If You Have A Base Class, Always Define A Virtual Dtor For It

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
  - This is because in construction or destruction, **when calling a virtual function, C++ does NOT use dynamic dispatch there**, and will only use the definition from the class where the constructor is defined. This is to **prevent calling a virtual function that touches uninitialized data.** See below example

```cpp
#include <iostream>

class Base {
public:
    Base() {
        display();
    }

    virtual ~Base() = default;

    virtual void display() const {
        std::cout << "Display from Base" << std::endl;
        
    }
    
    virtual void displayer_wrapper() const{
                display();
        }
};

class Derived : public Base {
public:
    Derived() {}

    ~Derived() override = default;

    void display() const override {
        std::cout << "Display from Derived" << std::endl;
    }
};

int main() {
    // See "Display from Base", because it's called in base ctor and polymorphism is banned at the point.
    Base* d = new Derived();
    // See "Display from Derived", which is expected from polymorphism
    d ->  displayer_wrapper();
    delete d;
    return 0;
}
```

- The specifier `final` can be used here to indicate that no child class can further overrding this function:

```cpp
virtual void foo() override final; 
```

    - `final`, and `override` should be in `.hpp`. Not in `.cpp`.

## Virtual Inheritance

The Dreadful Diamond on Derivation (DDD) Problem

TODO

## References

1. Stack Overflow explanation on Virtual inheritance: stackoverflow.com/a/21607/5652483
2. Wikipedia on Dynamic Dispatch: <https://en.wikipedia.org/wiki/Dynamic_dispatch>
