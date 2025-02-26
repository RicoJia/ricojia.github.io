---
layout: post
title: C++ - Lvalue and Rvalue Reference
date: '2023-01-10 13:19'
subtitle: Reference Lifetime
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Reference Lifetime 


1. const lvalue reference Simply Extends the Lifetime of A Temp. No copy constructor is called when the temp object is passed in
2. **const lvalue reference as a class attribute DOES NOT EXTEND the lifetime of the temp**. It simply stores a reference to it, and it's dangerous

```cpp
class MyClass {
public:
    // Default constructor
    MyClass() {
        std::cout << "Default constructor called\n";
    }
    // Copy constructor
    MyClass(const MyClass&) {
        std::cout << "Copy constructor called\n";
    }
    ~MyClass(){
        std::cout << "MyClass Dtor called\n";
    }
};
void funcByConstRef(const MyClass &obj) {
    std::cout << "Inside funcByConstRef\n";
}
class B{
    public:
    B(){std::cout << "B ctor\n"; }
    void foo(){ std::cout << "B foo\n"; }
    ~B(){ std::cout << "B dtor\n"; }
    const MyClass& m {};
};

int main() {
    // Passing by const reference: 
    funcByConstRef(MyClass());   // No copy is made, copy constructor is NOT called. The temporary object's lifetime is simply extended.
    
    std::cout << "==============================\n";
    B b;    // can see 'MyClass Dtor called' here. So the const lvalue reference is no longer valid!!
    b.foo();
    return 0;
}
```