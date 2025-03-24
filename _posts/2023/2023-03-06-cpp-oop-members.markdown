---
layout: post
title: C++ - [OOP] Members
date: '2023-03-06 13:19'
subtitle: Member Attributes and Methods, Copy, Move
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Member Attributes

Let's take a look at a tricky example

```cpp
#include <iostream>
#include <vector>
using namespace std;

class A {
public:
    int value;
    A(int v = 0) : value(v) {
        cout<<"Constructing A"<<endl;
    }
    // Copy constructor
    A(const A& other) : value(other.value) {
        std::cout << "A copied! value = " << value << std::endl;
    }
    // Move constructor
    A(A&& other) noexcept : value(other.value) {
        std::cout << "A moved! value = " << value << std::endl;
    }
};

class MyClass {
private:
    std::vector<A> data_ { A(1)};
public:
    MyClass() {cout<<"Constructing my class"<<endl;}

    // Return a const reference to data_
    const std::vector<A>& getData() const {
        cout<<"Getting data"<<endl;
        return data_;
    }
};

int main() {
    MyClass obj;
    std::cout << "Binding by const reference (no copies expected):" << std::endl;
    const auto& dataRef = obj.getData(); // No copies of A happen here
    std::cout << "\nBinding by value (copies will happen):" << std::endl;
    auto dataCopy = obj.getData(); // Copies A elements
    return 0;
}
```

What is the output?

```cpp
Constructing A
A copied! value = 1
Constructing my class
Binding by const reference (no copies expected):
Getting data

Binding by value (copies will happen):
Getting data
A copied! value = 1
```

Surprise? Here I have some pointers:

- Copy Happens In Temporary Construction In Initializer List
    - This is because **C++ standard specifies that elements in an initializer list is treated as a `const`**. Though we have a move contructor, to make the temporary a const, an extra copy needs to happen.
- We are able to return a **const reference to a member variable**. The receiver should specify that as well:
    ```cpp
    const std::vector<A>& getData() const {
        ...
    }
    const auto& dataRef = obj.getData();

    // This creates a copy
    auto dataCopy = obj.getData();
    ```