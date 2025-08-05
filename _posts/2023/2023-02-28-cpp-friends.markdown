---
layout: post
title: C++ - Friends
date: '2023-02-28 13:19'
subtitle: Friend Class, Friend Function, Inner Class, Forward Declaration
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

In C++, friend functions and friend classes provide controlled access to private members of a class. While useful, they should be used judiciously to maintain encapsulation and avoid unnecessary dependencies. Let's explore their roles and best practices.

## `friend` Functions

A friend function allows external functions to access the private and protected members of a class **without being a member function itself**. A friend function has the **same privilege level as a member function** but is not bound to a specific object instance.

### Best Practice: Place friend Functions at the Top of the Class Definition

Unlike member functions, friend functions do not require public or private access specifiers. A common convention is to place them at the beginning of the class definition, near the public section for clarity.

```cpp
class Foo {
    friend std::istream& operator>>(std::istream&, Foo&);
public:
    // Other members
};
```

### When a friend Function - Accessor, and Overloading

Sometimes, making a function friend is unnecessary. For example:

- Can I add `std::istream & operator>>(std::istream & is, Transform2D & tf);` as a friend function to Transform2D so I can modify its private Matrix values?

The answer: No, you don’t need to. In this case, a simple overloaded function can achieve the same result.

## `friend` Classes
 
A friend class can access all private and protected members of another class. This is useful in cases where a separate class requires intimate knowledge of another’s implementation. (So this is even more "intimate" than a child class)

Example: Granting Student_Handle Access to `Core`: by declaring Student_Handle as a friend of Core, all member functions of Student_Handle can access Core's private and protected members.

```cpp
class Core {
    friend class Student_Handle;
};
```

### Important Considerations

- Friendship is Not Mutual:
    - Unless explicitly specified, if Class A declares Class B as a friend, Class B does not automatically consider Class A a friend.
- Friendship is Not Inherited:
    - If Core has a friend class, **its derived classes do not automatically inherit this friendship**, unless explicitly declared.
- Friendship is Not Transitive:
    - A friend of a friend does not gain access. E.g., if Class A grants Class B access, Class B cannot pass this privilege to Class C.

### Forward Declaration

In reality, it's a common scenario where a friend and its friended classes live in two hpp files. In that case, we need to forward declare the friend class in the header file of the friended classes.

```cpp
// A.hpp  
#pragma once

// forward declaration only
class ESKF;

class IMUIntegrator
{
  // now you can friend it without pulling in the full definition
  friend class ESKF;

public:
  IMUIntegrator();
  void integrate();

protected:
  double dt_;
  // …
};

// B.hpp
#include "A.hpp"
class ESKF{
    
}
```

## Inner (Nested) Classes

A nested class is defined within another class, making it an inner class. **By default, an inner class is implicitly a friend of its enclosing class, meaning it can access private members.**

```cpp
class Enclosing {
private:
    int x = 30;  // Private member
public:
    class Nested {
    public:
        void nested_func(Enclosing* e) {
            cout << e->x;  // Allowed: Nested class can access private members of Enclosing
        }
    };
    Nested n_;  // Nested must be declared
    void func() {
        n_.nested_func(this);
    }
};
```

- Therefore, access modifiers do not matter: The inner class can be declared in the private, protected, or public section of the enclosing class—it still has access to private members.
- **The Nested Class Must Be Declared First**: To use a nested class inside the enclosing class, it must be declared before being instantiated.