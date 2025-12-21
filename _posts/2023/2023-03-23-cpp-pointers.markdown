---
layout: post
title: C++ - [Pointers - 1] - Raw Pointer
date: '2023-03-23 13:19'
subtitle: Raw Pointers, Array, Casting
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Raw Pointers

### 🚀 Basics

1. new and delete:

    - If you use new, you must use delete — otherwise, memory leak! [see code](https://github.com/RicoJia/notes/blob/master/examples/c%2B%2B_examples/dynamic.cpp).
    - `new[]`:**size must be provided as part of the signature**.

        ```cpp
        new int[10]();           // value-initialized to 0
        new int[5]{1, 2, 3};     // OK: extra elements zero-initialized
        new int[]{1, 2, 3};      // ❌ error: size must be explicit
        ```

    - **new[] must match delete[]**. A simple `delete` will yield undefined behaviour.
          - delete will take not into account the number of objects, but `delete[]` will.
          - delete[] should not be used on regular `char*`, unless it is ```char* c = new char[size_];```.

2. Structs that contain pointers
    - Default-initialized pointers inside a struct are `nullptr` — but that means no memory is allocated. → You must `new` those inner pointers manually.

### 📌 Using Pointers

1. Always initialize your pointers! Uninitialized pointers cause random crashes or data corruption.

```cpp
int* p;  // Not initialized yet
*p = 10; // dangling pointer, segfault!
// do this: 
int *q = nullptr
```

2. It's more often to use `int** ptr` than `int*& ptr`

```c++
void updatePtr(int*& ptr); // Uncommon
void updatePtr(int** ptr); // Common
```

- `int** ptr` can take an rvalue reference (&some_tmp_ptr), but `int *&` needs the ptr to be lvalue reference. so it's more flexible
- In C, people have been using `int**`

3. After `delete` or `free`, set the pointer to `nullptr`. This avoids use-after-free or double-free errors.

    ```cpp
    delete ptr_;
    ptr_ = nullptr;
    ```

4. Const type pointer is read-only pointer (const T pointer), while `type* const`  is const pointer to T

```c++
// main.cpp: In function ‘void process(const int*)’:
// main.cpp:13:10: error: assignment of read-only location ‘* ptr’
void process(const int* ptr) {
    *ptr = 9;
}

void keep_pointer_same_address(int* const ptr) {
    // ptr = ...;  // ❌ cannot change ptr
    *ptr = 5;      // ✅ can change the int
}
```

- It's recommended to add const specifier to raw pointers whenever possible.
- `int* const ptr` → `constant pointer to int`
- `const int* ptr` → `pointer to constant int`
- `const shared_ptr<T>` ≈ `T* const` → shared_ptr can't be reassigned, but pointee can be mutated
- `shared_ptr<const T>` → pointer to const object
  - If you're using `shared_ptr<const T>`, you must initialize in initializer list, especially for class members.

5. Do nullptr check unless you are 100% confident that the pointer is not Null

```c++
void (*fp)(int) = nullptr;
fp(42);  // 灾难！没检查就直接调用

// 或者更糟的情况
void (*fp)(int);  // 未初始化就使用
fp(42);  // 更大的灾难！
```

## Why Raw Pointer Is Not Recommended

1. Initialization:
 - Pointers are not auto-initialized.
 - Always explicitly initialize to nullptr in constructors or field declarations.
2. **Getting array size is unsafe**: `sizeof(int*) / sizeof(int);  // ❌ not safe`
3. Raw pointer fragility: shallow copy will point to the same memory address, which could be double-deleted.

```c++
class Resource {
    int* data;
public:
    Resource() { data = new int(42); }
    ~Resource() { delete data; }
    
    // Resource(const Resource& other) = default;  // shallow copy
    // Resource& operator=(const Resource& other) = default;  // shallow copy
};

void disasterExample() {
    Resource r1;
    Resource r2 = r1;
    // when the function terminates, r1 and r2 will delete the same pointer, disaster!
}
```

4. You must delete a pointer after use, and highly highly recommended to set it to **nullptr**

```c++
void noLeak() {
    int* p = new int(42);
    delete p;
    p = nullptr;  // 删除后最好置空
}
```

5. Not deleting an array using `delete[]`

```c++
int* arr = new int[10];
delete arr;    // ❌
```

6. **[ADVANCED]** Moving pointer that's going to be deleted. Usually, just create a new pointer.

```c++
int* p = new int[5];
for(int i = 0; i < 5; i++) {
    cout<<*p<<endl;
    p++;  // 
}
delete[] p;  // Nope, P has been moved. Just use a temp pointer
```

## When to use raw pointers

- Small, performance-critical code with clear ownership semantics.
- If your function doesn't care about ownership, prefer:

    ```cpp
    void f(widget& w);     // ✅ clear, safe
    void f(widget* w);     // ✅ OK if nullable
    void f(shared_ptr<widget>& w); // ❌ misleading — might reset or alias others
    ```

### 📚 Arrays

- `arr[1] = 10;` will compile even if arr doesn’t own the memory. → Might segfault at runtime depending on what arr points to.
- Multidimensional arrays:
  - For `arr[1][2]` to work, arr must be `int**`, not `int*`.
  - Or use `std::vector<std::vector<int>>` for safety.
  - No clean way to convert pointer back to array form:
    - Raw pointers have no size metadata.
    - You lose bounds-checking and iteration safety.

## Casting

`static_pointer_cast` vs `dynamic_pointer_cast`:

- ```RTTI``` is run time type information, if a base class **has at least a virtual function**, then base class ptr can be **dynamic_cast** to derived class
  - **(downcast, upcast is derived -> base)**
  - ```dynamic_cast``` is only for downcasting. dynamic_cast can happen in runtime.
  - **If downcasting fails (not derived class), nullptr is returned.**
  - You can downcast references as well → throws `std::bad_cast`
  - Safer, but slower than `static_cast`

- ```static_cast``` happens **during compile time, no RTTI is needed**.
  - **If you're sure you can downcast to a specific type**, use static_cast since it's cheaper, and the language allows you to do so

- ```static_pointer_cast``` vs ```static_cast```: ```static_pointer_cast``` works on shared_ptrs, because you can't cast its type directly.

    ```cpp
    #include <iostream>
    using namespace std;
    class B {
        //virtual void fun() {} // NEED TO ADD THIS!!
    };
    class D : public B {
    };

    int main()
    {
        B* b = new D;
        D* d = dynamic_cast<D*>(b);

        // 1. use dynamic_cast if we're not sure if we can succeed
        if (d != NULL)
            cout << "works";
        else
            cout << "cannot cast B* to D*";

        // 2. cpp still allows you to use static_ptr_cast
        std::static_pointer_cast<DerivedClass>(ptr_to_base)->f();
        // 3. even static_pointer_cast
        static_cast<DerivedClass*>(ptr_to_base.get())->f();      // equivalent to above

        return 0;
    }
    ```
