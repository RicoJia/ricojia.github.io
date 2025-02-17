---
layout: post
title: C++ - [OOP] Polymorphism
date: '2023-03-03 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Virtual Functions

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
