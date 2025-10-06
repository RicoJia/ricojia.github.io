---
layout: post
title: C++ - [Templates 3] Template Deduction, Instantiation
date: '2023-02-12 13:19'
subtitle: Template Argument Deduction
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Template Argument Deduction

Direct template argument substitution works! So template argument reduction could work in this case.

```cpp
template <class LidarMsg>
void process_pt_cloud(const std::shared_ptr<LidarMsg>& msg);
sensor_msgs::msg::PointCloud2::SharedPtr pc2_msg;   // std::shared_ptr<sensor_msgs::msg::PointCloud2>
process_pt_cloud(pc2_msg);

// Fine, LidarMsg = sensor_msgs::msg::PointCloud2 ✅
```

A qualified-id is not deducible in templates, because the compiler needs to "reverse-engineer" to check `LidarMsg`

```cpp
template <class LidarMsg>
void process_pt_cloud(const typename LidarMsg::SharedPtr& msg);

using SharedPtr = std::shared_ptr<PointCloud2>;

sensor_msgs::msg::PointCloud2::SharedPtr pc2_msg;
process_pt_cloud(pc2_msg);
```

In this case, `?::SharedPtr = sensor_msgs::msg::PointCloud2::SharedPtr`. The compiler is NOT smart enough to look that up reversely.

## Template Instantiation

## Template Instantiation

### Free (non-member) Template Functions

```cpp
template <typename T>
void foo(T x) { ... }
```

When you call `foo(42)`, the compiler needs to instantiate `foo<int>` and this requires the definition of `foo<T>` to be visible at the point of calling. If you define the function in a `.cpp` file, the definition there is not an instantiation for other TUs to see. Therefore, template functions **must be instantiated in header files**

### Template Classes

Member functions in a template class suffer the same issue. Then need to be instantiated in a header file so other functions can see them. One mitigation is to instantiate them at the bottom of a `.cpp` file, **but that's non-standard**.

```cpp
// .hpp
template <typename T>
class Bar {
public:
    void baz();
};

// .cpp
#include "bar.hpp"

template <typename T>
void Bar<T>::baz() {
    // body
}
template class Bar<int>;
```

Therefore, it's nice to define these functions in the same `.hpp` file.
