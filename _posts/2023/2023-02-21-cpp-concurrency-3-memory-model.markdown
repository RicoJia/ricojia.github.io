---
layout: post
title: C++ - [Concurrency 3] Memory Model
date: '2023-02-21 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## Introduction

The C++ memory model was formally introduced in **C++11**  mainly for multithreading. Before C++11:

- Threading was platform-dependent. (POSIX threads for Unit systems)
- Behavior of shared memory and data races **was undefined and left up to the OS/hardware/compiler.**
    - 🫠️ "Pthreads lib assume no data race." Pthreads are a **thin wrapper over OS-level primitives** and don’t prevent data races. It's the programmer’s job to synchronize correctly.

C++11 added:

- A standard memory model.
    - Compared to `Pthreads`, the C++11 memory model enforces stricter semantics, which helps correctness but might introduce overhead compared to bare-metal threading APIs like Pthreads.
    - ❗️ "Memory ordering: Thou shalt not modify the behavior of a single-threaded program". The compiler is **free to reorder instructions for optimization as long as the observable behavior of a single-threaded program doesn’t change** — this is called the as-if rule. But in multithreaded programs, these reorderings can cause data races unless explicitly synchronized, which is why C++11 introduced a well-defined memory model.
- `std::thread, std::mutex, std::atomic` types

This brought C++ more in line with Java/C# in terms of native threading support.

## Example: Memory Reordering by the Compiler

C++ compilers can reorder instructions as part of optimization — as long as the observable behavior in a single-threaded program is preserved (per the as-if rule). This can lead to subtle issues in multi-threaded programs.


[source](https://preshing.com/20120625/memory-ordering-at-compile-time/)

- cpp code
    ```cpp
    int A, B;
    void foo()
    {
        A = B + 1;
        B = 0;
    }
    ```

- without optimization: 
    ```assembly
    $ gcc -S -masm=intel foo.c
    $ cat foo.s
    ...
    mov     eax, DWORD PTR _B  (redo this at home...)
    add     eax, 1
    mov     DWORD PTR _A, eax
    mov     DWORD PTR _B, 0
    ...
    ```

- With Optimization: 
    ```cpp
    $ gcc -O2 -S -masm=intel foo.c
    $ cat foo.s
    ...
    mov     eax, DWORD PTR B
    mov     DWORD PTR B, 0
    add     eax, 1
    mov     DWORD PTR A, eax
    ...
    ```

**Why this matters**: One can see that it's possible that B (which may be atomic) can be ready before A (which may/may not be atomic). In a multithreaded context, this reordering can cause issues. For example, another thread might observe B == 0 while A has not yet been updated. This can violate intended synchronization logic.

**How to prevent:** Use `std::atomic` with proper memory ordering (e.g., `memory_order_seq_cst`) to prevent unwanted reordering.

Atomics act as compiler and CPU fences, ensuring ordering constraints are respected where required.

## C++ Memory Model vs Lower Level Register Reads:

In a low-level device register, a read is usually read-clear (reading an IRQ status bit), or read-toggle. Two cores reading simultaneously could acknowledge it twice. 

Some dangerous MMIO (Memory-Mapped IO)examples include:

| Scenario (MMIO)                               | Safe? | Why / Remedy                   |
|----------------------------------------------|-------|--------------------------------|
| Two threads poll a PCIe status register      | ❌    | Read‑clear → lost/dupe events. Use a single polling thread or a spin‑lock. |
| CPU reads a continuously updating 64‑bit timer| ⚠️    | Possible tear. Follow the ‘latch‑high‑then‑low’ sequence in the datasheet. |
| Two cores read a ROM device‑ID register      | ✅    | No side‑effects. Still mark as `volatile` / use `ioread32()`. |

The above is **well outside the C++ memory model**. The C++ memory model explicitly excludes "actions performed by or on behalf of the hardware". Correctness is platform specific. 

By the C++ memory model:

- Concurrent read-only access is thread-safe, **iff no other threads are modifying**
- Read / write to 8 byte `std::atomic<T>` is atomic, no guarantee to regular objects

**Rule of thumb**: Ordinary RAM objects obey the C++ memory model; MMIO obeys the hardware datasheet + architecture I/O ordering rules.


