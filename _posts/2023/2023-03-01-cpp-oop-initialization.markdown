---
layout: post
title: C++ - [OOP] Initialization
date: '2023-01-15 13:19'
subtitle: Default Initialization, Memory Order
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Default Initialization

Default Initialization is **when a variable is initialized without explicit value**. They could be in an uninitialized state and accessing them could be an undefined behavior.

- POD types like `int`, `float` are initialized to an **intermediate value**. Accessing them without initialization is an undefined behavior,
    - Default initialization guarantees to initialize POD data, **but not array data**
- User defined types should either **have no constructor provided, or have a default constructor present.**

```cpp
class A{
    A(){};  // This is NOT default initialization
    A() = default;  // This IS default initialization
};

class B{
    int value;  // There's a synthesized default initializer as well.
};
```

- `static` storage duration always default initializes to 0, whereas automatic storage variables are default initialized to intermediate values

```
int x;          // automatic storage, intermediate value
static int y;   // default initialized to 0
int *j;         // automatic storage, intermediate value
static int *p   // static storage, default initialized to nullptr

int arr[5];        // Elements have indeterminate values
static int arr[5]; // All elements default-initialized to 0
```

Best Practices:

- Use **brace initialization** `{}` (C++ 11) to explicitly initialize variables

```cpp
int i{};    // explicitly initialized to 0;
```

## Initialization Order

We need to make sure the vairable order is consisntent in both the initializer list and the variable definition. One common warning we see is `warning: <VAR> will be initialized after [-Wreorder]`

```cpp
class MyClass{
    /*
        * Below public functions are defined in the order of operation
    */
    public:
        // Step 0: Create this object when first powered on
        MyClass(): previous_time_(millis()), current_status_(Status::UNINITIALIZED)
        {}

        Status current_status_;
        unsigned long previous_time_;
};
```

- The order of class variable initialization is determined by the declaration order, not the order in initializers. [Reference](https://wiki.sei.cmu.edu/confluence/display/cplusplus/OOP53-CPP.+Write+constructor+member+initializers+in+the+canonical+order)

### Memory Order of Class Members

Declaration Order: Variables appear in memory in the order they are declared in the struct/class.

```cpp
struct Example {
    int a;     // Offset 0
    double b;  // Offset 8 (if 8-byte alignment is required for double)
    char c;    // Offset 16 (next available slot for alignment)
};
```

### Default Parameters

- Default values are given in `.hpp` files, not `.cpp`. In `.hpp` file, do `void VoxelGrid(CloudPtr cloud, float voxel_size = 0.05);`. You wouldn't need it in the `.cpp` files