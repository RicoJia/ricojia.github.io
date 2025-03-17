---
layout: post
title: C++ - Sizing and Type Conversion
date: '2023-01-17 13:19'
subtitle: Memory Alignment, Sizeof
comments: true
header-img: "img/post-bg-alitrip.jpg"
tags:
    - C++
---

## Memory Alignment

C++ struct / class does memory alignment. Here's an illustrative example:

```cpp
struct LocalizationData {
    float x, y;  // Each float is 4 bytes
    bool r;      // bool is 1 byte
};
sizeof(LocalizationData);   // see 12 bytes, without memory alignment, it should be 9 bytes
```

- float (4 bytes): Typically requires 4-byte alignment.
- bool (1 byte): Typically requires 1-byte alignment, but it can also be padded to match the alignment of the structure.

With alignment, and with proper offsets, we can access data in a more modular way. Otherwise, there could be performance penalties.

<div style="text-align: center;">
<p align="center">
    <figure>
        <img src="https://github.com/user-attachments/assets/3df8bf6e-d246-4609-b449-eb3faa51b810" height="300" alt=""/>
        <figcaption><a href="https://ncmiller.dev/memory-alignment.html">Source</a></figcaption>
    </figure>
</p>
</div>

To enable and disable memory alignment, one can:

```cpp
#include <iostream>
using namespace std;
// remove memory alignment of bool and float. otherwise, more memory is padded to bool
#pragma pack(1)
struct LocalizationData {
    float x, y;
    bool r;
};
#pragma pack()  // Restore default alignment
struct LocalizationData2 {
    float x, y;
    bool r;
};
int main()
{
    std::cout<<sizeof(LocalizationData)<<std::endl;    // see 9 because memory alignment is disabled

    std::cout<<sizeof(LocalizationData2)<<std::endl;    // see 12 because memory alignment is enabled

    auto d2 = LocalizationData2();
    std::cout<<sizeof(d2.r)<<std::endl;                 // see 1
    
    return 0;
}

```

Alignment and Padding:

- To satisfy alignment requirements, compilers may add padding between variables or at the end of the struct.
- For example, in the struct above, a (4 bytes) might leave a 4-byte padding before b (8 bytes) to align b on an 8-byte boundary.
- Inheritance: for classes with inheritance, memory layout may include hidden padding or vtable pointers (if virtual functions are used). For non-inherited plain structs, the layout is straightforward.


### Miscellaneous Size Issues

-  Does static constexpr Change the Object's Size?
    - No. `static constexpr` variables belong to the class itself, not part of the instance. E.g, 
        ```cpp
        struct Example {
            static constexpr int static_var = 42;  // Shared by all objects, not part of any instance
            int a;
        };  // 4bytes, only for a
        ```

- `sizeof(object_or_type)` returns the number of bytes a type or object consumes. 

## Type Conversion

### `reinterpret_cast` vs `static_cast`

- `reinterpret_cast<char*>(unsigned_char_ptr)` vs `static_cast<char*>(unsigned_char_ptr)` (which doesn't work)
    - `static_cast` converts types where conversion is well defined. But it does NOT support conversions between unrelated pointer types. `char*` vs `unsigned char*`
    - Low-Level Pointer Conversion: reinterpret_cast allows arbitrary type conversion between pointer types, essentially bypassing type safety. **USE WITH CAUTION**

### [Case Study] Mixing size_t and int: Potential Pitfalls and Best Practices

In C++, mixing `size_t` (typically an unsigned 64-bit integer) with `int` (usually a signed 32-bit integer) can lead to unexpected behavior due to implicit type conversions. Key Issues:

- Negative int converts to a large size_t:
    - If a negative int is implicitly converted to size_t, it wraps around to a very large positive value.

```
size_t s = -1;  // -1 becomes 18446744073709551615 (UINT64_MAX on a 64-bit system)
```

- Loop Conditions (i < size_t): A loop like for `(int i = n; i >= 0; --i)` can become an infinite loop if n is size_t, since i will never be negative in unsigned comparison.

Best Practices:

✅ Use `ssize_t` instead of size_t for signed indices. `ssize_t` is the signed equivalent of size_t, preventing unintended conversion issues.

```cpp
ssize_t index = -1;  // Safe, avoids unsigned wrapping issues
```


    