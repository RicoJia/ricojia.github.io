---
layout: post
title: C++ - Language Properties
date: '2023-02-16 13:19'
subtitle: Zero-Overhead Abstraction, Garbage-Collection-Free, Endianness Handling, One-Definition-Rule (ODR)
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

C++ and C are known for their high performance. High level languages like Python are known for their relative ease of use. In this article, we will compare the two sides and see what makes C++ special (for its high performance, and lower convenience of use)

## Zero Overhead Abstraction

In C++, non-virtual function calls have zero runtime overhead because they are resolved at compile time. However, virtual functions introduce a small performance cost due to vtable lookups and the use of a vptr (virtual table pointer). In contrast, Python treats every function as an object, meaning function calls always involve an extra level of indirection, regardless of overriding / non-overriding methods. 

Here, one example is that heap allocation is always a few more cycles more expensive than stack allocation. That is illustrated in the example below:

- In Stack Allocation:

```cpp
int fun2() {
    int x = 3;
    return x;
}
```
Its assembly shows that we simply put the value `3` onto the stack. Easy peasy. With C++20's `consteval`, this can be used for [further optimizations](https://ricojia.github.io/2023/02/01/cpp-constness/)

```cpp
fun2():
    push    rbp
    mov     rbp, rsp
    mov     DWORD PTR [rbp-4], 3
    mov     eax, DWORD PTR [rbp-4]
    pop     rbp
    ret
```

- In heap allocation:

```cpp
int fun1() {
    int* x = new int(3);
    return *x;
}
```

Associated assembly code involves: creating a pointer on stack, calling `operator new` in runtime to allocate memory (so it can't be optimized during compile time), assign variable, and return the dereferenced result

```cpp
fun1():
    push    rbp
    mov     rbp, rsp
    sub     rsp, 16
    mov     edi, 4
    call    operator new(unsigned long)
    mov     DWORD PTR [rax], 3
    mov     QWORD PTR [rbp-8], rax
    mov     rax, QWORD PTR [rbp-8]
    mov     eax, DWORD PTR [rax]
    leave
    ret
```

**In comparison, Python almost allocates EVERYTHING on the heap**. That slows down memory allocation.


## Garbage Collection and Performance in C++ vs. Python

C++ does not require garbage collection because it relies on deterministic object destruction. When an **exception is thrown inside a try block, all objects within the block are immediately destructed in reverse order of their creation**, ensuring resource cleanup. This mechanism, **known as RAII (Resource Acquisition Is Initialization)**, provides predictable performance **without the overhead of garbage collection.**

In contrast, Python uses automatic garbage collection with **reference counting and periodic garbage collection sweeps**. If an exception occurs **in a try block, objects may not be immediately destroyed**, especially if **there are circular references or if the garbage collector has not yet reclaimed them**. This can lead to delayed resource cleanup and increased memory usage until the garbage collector runs.

## Endianness Enforcement

Endianness refers to the byte order used to store multi-byte data types in memory. There are two main types:

- Big-endian: Stores the most significant byte (MSB) first.
- Little-endian: Stores the least significant byte (LSB) first.

**C++ does not enforce a specific endianness**; it is determined by the underlying hardware architecture. For example:

- x86 and ARM (in most configurations) use **little-endian**.
- Some older architectures (e.g., Motorola 68k) use big-endian.

This means that code written in C++ needs to be aware of potential endianness mismatches when **communicating between different systems** (e.g., **network protocols or file formats**).

```cpp
bool is_little_endian(){
    // check if LSB is stored first:
    int num = 1;
    return *(char*)(&num) == 1;
}
int main() {
    if (is_little_endian()) {
        std::cout << "Little-endian\n";
    } else {
        std::cout << "Big-endian\n";
    }
    return 0;
}
```

### Endianness In Various Systems

Java abstracts away endianness by specifying a standard for certain operations. For example, Java's `DataInputStream` and `DataOutputStream` use big-endian by default. 

ROS handles endianness internally in message serialization and deserialization to ensure compatibility, but developers must still be cautious when working with custom binary formats.

AVR-based Arduinos (e.g., ATmega328P on Arduino Uno, Mega), and ESP boards use Little-Endian.

Raspberry Pi (RPi) use ARM-based processors, and they all are little-endian by default. 
    - We still need cross-compilationdue to differences in **instruction set architecture (ISA)** and **system libraries.**
    - Raspberry Pi uses ARM-Linux (e.g., ARM Cortex-A series). Most PCs use x86/x86-64 (e.g., Intel, AMD).

Python **does not enforce a specific endianness at the language level**. Instead, it provides tools to handle endianness when working with binary data.

- Integers in Python are **arbitrary-precision**, meaning they are not stored in a **fixed-size memory representation like C/C++ integers**. They can grow as large as needed.
    - Python’s int type is implemented using the `PyLongObject` structure in `C` (defined in `CPython`). It consists of:
        - Sign – Determines if the number is positive or negative.
        - Digit Array – Stores the actual number in base 2³⁰ or 2³¹, depending on the platform.
        - Size – The number of "digits" used in the array.
        ```c
        struct _longobject {
            Py_ssize_t ob_refcnt;  // Reference count
            struct _typeobject *ob_type;  // Type info
            Py_ssize_t ob_size;  // Number of digits (negative if negative number)
            uint32_t ob_digit[1];  // Array of digits (stores the number)
        };
        ```
    - So the number `1234567890123456789` is`0x112210F4B16C1B05` in C (8 bytes). In python, it would be stored as 2 30-bit digits `0110110010110001100000110000101  (Least significant)` (4 bytes). Plus, there are: 24 bytes (as `PyLongObject` overhead in a 64-bit system). So in total, Python needs 32 bytes for storing this integer.
- When dealing with binary formats (e.g., file I/O, network communication), Python allows you to specify endianness using modules like `struct` and `int.to_bytes()`.

## One-Definition-Rule (ODR)

ODR-use ("one-definition-rule use") of an object requires  to **have a unique definition** in the program. Typical examples include:

- Taking the address of the object.
- Binding a reference to it.
- Using it in a context where a pointer to the object is needed.

**In a class**, if a static member is only used as a compile-time constant (e.g., in constant expressions), it may not require a definition. But if you take its address, **then it is odr-used**, and the linker needs to see a definition somewhere.


- In C++ 11 and 14, `static constexpr`, if we have an ODR use of the class member, **like getting a pointer to it**, we need to define it outside of the class with:

```cpp
constexpr int Foo::value;  // Out-of-line definition needed because of odr-use.
```

- In **C++17**, `static constexpr` members **are implicitly inline**, (`inline` is never needed for `static constexpr` class members), meaning they can be defined in the header without causing multiple definition errors when included in several translation units.

```cpp
#include <iostream>

struct Foo {
    static constexpr int value = 10;
};

// THIS LINE IS NEEDED
// constexpr int Foo::value;  // Out-of-line definition needed because of odr-use.

int main() {
    // ODR-use: taking the address of Foo::value.
    const int* ptr = &Foo::value;  
    std::cout << *ptr << std::endl;
    return 0;
}
```

